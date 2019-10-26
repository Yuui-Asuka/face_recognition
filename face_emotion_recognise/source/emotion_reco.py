import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import operator
import xgboost as xgb
import time
from scipy import misc
from PIL import Image
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from numpy import *
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from skimage import transform,data
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
np.set_printoptions(threshold=np.inf)

class Emotion:
    """train_path_fix : 该文件夹是一对多方式的训练数据，该文件夹下存在四个文件，每个文件的内容如下：
                      fix1 : [类别0] [类别123混合]
                      fix2 : [类别1] [类别023混合]
                      fix3 : [类别2] [类别013混合]
                      fix4 : [类别3] [类别012混合]

       train_path : 该文件夹是训练一个模型时的训练数据，分为四个类别：
                      [0] : 高兴，[1] : 不高兴，[2] : 惊讶，[3] : 愤怒

       model_path : 该文件夹是训练模型存放的文件夹，载入模型时也会从该文件夹中载入

       image_size : 该参数是图片的尺寸，与训练数据所用的尺寸保持一致
    """
    def __init__(self,train_path_fix,train_path,test_path,model_path,image_size):
        self.filters = self.build_filters()
        self.train_path_fix = train_path_fix
        self.train_path = train_path
        self.test_path = test_path
        self.model_path = model_path
        self.image_size = image_size

    def build_filters(self):
        """该方法用于构建gabor滤波器"""
        ksize = [1,2,3]                                      # gabor尺度
        lamda = np.pi / 2.0                                  # 波长
        filters = []
        for theta in np.arange(0, np.pi*15/8, np.pi/8):      # gabor方向，0°，45°，90°，135°，共6个
            for K in range(3):
                kern = cv2.getGaborKernel((ksize[K], ksize[K]), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
                kern /= 1.5 * kern.sum()
                filters.append(kern)
        return filters

    def process(self,img,filters):
        """滤波过程"""
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
        return accum


    def getGabor(self,img,filters):
        if (img.ndim == 2):                                  # 判断输入的图是否为灰度图
            im_gray = cv2.equalizeHist(img)                  # 灰度图像直方图均衡化
            img_ndarray = np.asarray(im_gray)
        else:
            im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 进行灰度均衡化
            im_gray = cv2.equalizeHist(im_gray)              # 灰度图像直方图均衡化
            img_ndarray = np.asarray(im_gray)
        res = []                                             # 滤波结果
        norm = []                                            # 范数结果
        (x,y)=img_ndarray.shape
        for i in range(len(filters)):                        # 取一个图片的特征
            res1 = self.process(img_ndarray , filters[i])
            res.append(res1)
        for i in range(x):                                   # 取每个像素值的特征的L2范数
            for j in range(y):
                res = np.asarray(res)
                res2=res[:,i,j]
                res3 = np.linalg.norm(res2, ord=2)           # 计算矩阵res1的范数
                norm.append(res3)
        res = np.asarray(norm)
        return res

    def load_training_data(self,imagePath,random_split = False):
        """两种方式载入数据，一种是将测试集和训练集放在两个不同的文件中，分别载入；
        另一种是将所有的数据放在同一个文件夹下，按照比例随机切分"""
        label=[]
        total=[]
        if not os.path.isdir(imagePath):
            raise IOError("The folder " + imagePath + " doesn't exist")

        for root, dirs, files in os.walk(imagePath):
            for filename in (x for x in files if x.endswith(('.tiff','.jpg','.png'))):
                filepath = os.path.join(root, filename)
                object_class = filepath.split(os.sep)[-2]     # 情绪标签
                if object_class == '0':
                    label.append(0)
                elif object_class == '1':
                    label.append(1)
                elif object_class == '2':
                    label.append(2)
                elif object_class == '3':
                    label.append(3)
                elif object_class == '012':
                    label.append(120)
                elif object_class == '013':
                    label.append(130)
                elif object_class == '023':
                    label.append(230)
                else:
                    label.append(123)
                image = np.array(Image.open(filepath))
                newData = self.getGabor(image,self.filters)    # 对图片进行特征提取
                total.append(newData)
        total = np.array(total)
        print(total.shape)
        if random_split == False:
            X = total[:,:]
            y = np.squeeze(label)
            y = y.reshape((-1, 1))
            return X,y
        else:
            X_train, X_test, y_train, y_test = train_test_split(total, label, test_size=0.1, random_state=0)
            return X_train,X_test,y_train,y_test

    def load_model(self,X,y,cla):
        if cla == 'xgb':
            clf = xgb.XGBClassifier(silent=False,
                            learning_rate = 0.07,
                            n_estimators=160,
                            colsample_bytree = 0.7,
                            max_depth = 5,
                            min_child_weight = 5,
                            gamma = 0,
                            reg_lambda = 0.0125,
                            subsample = 0.8)
        elif cla == 'linear':
            clf = svm.SVC(kernel='linear', C = 0.1)
        elif cla == 'rbf':
            clf = svm.SVC(kernel = 'rbf', C = 0.1, gamma = 0.0001)
        elif cla == 'poly':
            clf = svm.SVC(kernel = 'poly',degree = 6,gamma = 0.0001,C = 0.5)
        elif cla == 'dtree':
            clf = DecisionTreeClassifier(criterion = 'gini',splitter = 'random' ,max_depth = 20,max_leaf_nodes = 50)
        elif cla == 'randomforest':
            clf = RandomForestClassifier(n_estimators = 140 , criterion = 'entropy' ,n_jobs = -1)
        model = clf.fit(X, y)
        return model

    def feature_visualization(self):
        """该方法用于特征的可视化，可以看到各个特征对于决策重要性的排序"""
        X_train,X_test,y_train,y_test = load_training_data(self.train_path,random_split = True)
        params = {'booster':'gbtree','binary':'logistic','eta':0.07,'max_depth':4,'gamma':0,'silent':False,'n_estimators':160}
        xgtrain = xgb.DMatrix(X_train,label = y_train)
        bst = xgb.train(params,xgtrain,100)
        importance = bst.get_fscore()
        importance = sorted(importance.items(),key = operator.itemgetter(1))
        print(importance)
        df = pd.DataFrame(importance,columns = ['feature','fscore'])
        df['fscore'] = df['fscore']/df['fscore'].sum()
        plt.figure()
        df.plot(kind = 'barh',x = 'feature',y = 'fscore')
        plt.show()

    def train_clf_model(self):
        """该方法采用一对多的方式，将数据集分为两类：
        [0],[1,2,3]
        [1],[0,2,3]
        [2],[0,1,3]
        [3],[0,1,2]
        这四种分类方式各自训练一个模型，总共四个模型。"""
        paths = os.listdir(self.train_path_fix)
        start_time = time.time()
        for i,path in enumerate(paths):
            path = os.path.join(self.train_path_fix,path)
            i += 1
            print("正在加载第%d个模型的数据..." % i)
            train_x,train_y = self.load_training_data(path,random_split = False)
            print("正在训练第%d个XGB模型..." % i )
            model = self.load_model(train_x,train_y, cla = 'xgb')
            clf_model_path = os.path.join(self.model_path,("clf_model_" + "%04d"%int(i) + ".pkl"))
            print("正在保存第%d个模型..." % i )
            joblib.dump(model,clf_model_path)
        cost_time = (time.time() - start_time)
        print("模型训练完成！用时%4.5f秒。" % cost_time)

    def train_test_clf_model_2(self,**kwargs):
        """该方法使用一种分类器直接对4种表情进行分类，
        可以选择分类器为三种不同核函数的svm、决策树、随机森林和xgboost"""
        for key,value in kwargs.items():
            if value not in ['xgb','linear','rbf','poly','dtree','randomforest']:
               raise ValueError("please choose a portable classifier in 'linear','rbf','poly','dtree','randomforest'!")
        if len(kwargs) == 0:
            cla = 'xgb'
        else:
            cla = kwargs['classifier']
        path = self.train_path
        start_time = time.time()
        print("正在载入训练数据...")
        train_x,test_x,train_y,test_y = self.load_training_data(path,random_split = True)
        print("数据载入完毕，用时%4.5f秒。"% (time.time()-start_time))
        start_time = time.time()
        print("正在训练模型...")
        model = self.load_model(train_x,train_y,cla)
        clf_model_path = os.path.join(self.model_path,("clf_model.pkl"))
        print("模型训练完毕！用时%4.5f秒，正在保存模型..." % (time.time()-start_time))
        joblib.dump(model,clf_model_path)
        print("正在评估模型...")
        pred_y = model.predict(test_x)
        print("召回率和预测准确度:"+"\n")
        print(classification_report(test_y, pred_y, target_names=['0','1','2','3']))
        print("混淆矩阵：")
        print(confusion_matrix(test_y, pred_y, labels=range(4)))
        print(str(pred_y) + "\n" + str(test_y))
        acc_train = model.score(train_x,train_y)
        acc_test = model.score(test_x,test_y)
        print(model.__class__.__name__, "训练集预测精度:", acc_train)
        print(model.__class__.__name__, "测试集预测精度:", acc_test)

    def test_model(self):
        """该方法用于测试一对多的模型效果，将一张图片分别输入四个模型得到四个概率值，
        选取概率值最大的模型对应的类别作为分类的结果。"""
        print("正在载入模型...")
        model = joblib.load(r".\source\model\clf_model_0001.pkl")
        model2 = joblib.load(r".\source\model\clf_model_0002.pkl")
        model3 = joblib.load(r".\source\model\clf_model_0003.pkl")
        model4 = joblib.load(r".\source\model\clf_model_0004.pkl")
        print("\n" + "正在载入测试数据...")
        X, y = self.load_training_data(self.test_path)
        y_pred = model.predict_proba(X)
        y_pred2 = model2.predict_proba(X)
        y_pred3 = model3.predict_proba(X)
        y_pred4 = model4.predict_proba(X)
        feature = []
        for i in range(y_pred.shape[0]):
            photo = np.array([y_pred[i,0],y_pred2[i,0],y_pred3[i,0],y_pred4[i,0]])
            feature.append(photo)
        index = np.argmax(np.array(feature),axis = 1)
        print("召回率和准确度：")
        print(classification_report(y.ravel(), index, target_names=['0','1','2','3']))
        print("混淆矩阵：")
        print(confusion_matrix(y.ravel(), index, labels=range(4)))
        with open(r'add','a') as f:
            f.write(str(classification_report(y.ravel(), index, target_names=['0','1','2','3'])))
            f.write(str(confusion_matrix(y.ravel(), index, labels=range(4))))
        print(str(index) + "\n" + str(y.ravel()))

    def image_recognise(self,images,use_multi = False):
        """该方法用于图片标注"""
        total = []
        for i,image in enumerate(images):
            resized = misc.imresize(image, (self.image_size, self.image_size), interp='bilinear')
            newData = self.getGabor(resized,self.filters)
            total.append(newData)
        total = np.array(total)
        X = total[:,:]
        if use_multi == False:
            model = joblib.load(r".\source\model\clf_model.pkl")
            predict = model.predict(X)
        else:
            model = joblib.load(r".\source\model\clf_model_0001.pkl")
            model2 = joblib.load(r".\source\model\clf_model_0002.pkl")
            model3 = joblib.load(r".\source\model\clf_model_0003.pkl")
            model4 = joblib.load(r".\source\model\clf_model_0004.pkl")
            y_pred = model.predict_proba(X)
            y_pred2 = model2.predict_proba(X)
            y_pred3 = model3.predict_proba(X)
            y_pred4 = model4.predict_proba(X)
            feature = []
            for i in range(y_pred.shape[0]):
                photo = np.array([y_pred[i,0],y_pred2[i,0],y_pred3[i,0],y_pred4[i,0]])
                feature.append(photo)
            index = np.argmax(np.array(feature),axis = 1)
            predict = index
        emotion_dict = {0:'happy',1:'unhappy',2:'surprise',3:'angry'}
        emotion_list = [emotion_dict[p] for p in predict]
        return emotion_list
