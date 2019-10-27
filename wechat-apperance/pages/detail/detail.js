// pages/detail/detail.js
Page({

  /**
   * 页面的初始数据
   */
  data: {
    headimg:[
    { url: '/imgs/0001.png'},
    { url: '/imgs/0002.png'},
    ],
    details : null,
    src : null
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    this.ctx = wx.createCameraContext()
  },
  takePhoto() {
    this.ctx.takePhoto({
      quality: 'high',
      success: (res) => {
         this.setData({
         src : res.tempImagePath
         })
        var that = this 
        wx.uploadFile({
          url: 'http://192.168.43.31/people',
          // url: getApp.data.servers,
          filePath: that.data.src,
          name: 'file',
          formData: {
            user: 'test'
          },
          header: {
            "Content-Type": "multipart/form-data"
          },
          success(res) { }
        })
      }
    })
  },

  choiceImage(){
  wx.chooseImage({
    count: 1,
    sizeType: ['original', 'compressed'],
    sourceType: ['album', 'camera'],
    success(res) {
      // tempFilePath可以作为img标签的src属性显示图片
      const tempFilePaths = res.tempFilePaths
      wx.showToast({
        title: '正在上传...',
        icon: 'loading',
        mask: true,
        duration: 10000
      })  
      wx.uploadFile({
        url: 'http://192.168.43.31/people' , 
        // url: getApp.data.servers,
        filePath: tempFilePaths[0],
        name: 'file',
        formData: {
          user: 'test'
        },
        header: {
          "Content-Type": "multipart/form-data"
        },  
        success(res) {
        }
      })
    }
   
  })
  },
  error(e) {
    console.log(e.detail)
  },

  bindTextAreaBlur: function (e) {
    console.log(e.detail.value);
    var that = this;
    that.setData({
      details: e.detail.value

    });
    // console.log(that.data.details)
  },
  submit() {
    var that =this
     console.log(that.data.details)
    wx.request({
      url: 'http://192.168.43.31/people',
      data: {
      hhh: JSON.stringify(that.data.details)
        // name:'as'
      },
      method: "POST",
      header: {
        'content-type': 'application/x-www-form-urlencoded',
        'chartset': 'utf-8'
      }
    })
  },
transform(){
wx.navigateTo({
  url: '../recognise/recognise',
})
},

clear(){
    var that = this
    that.setData({
      details: ''
    });
  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function () {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {

  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide: function () {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload: function () {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function () {

  }
})
