// pages/recognise/recognise.js
Page({

  /**
   * 页面的初始数据
   */
  data: {
    headimg: [
      { url: '/imgs/0001.png' },
      { url: '/imgs/0002.png' },
      
    ],
    imageSrc: null,
    filePath2:null

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
          src: res.tempImagePath
        })
        var that = this
        wx.uploadFile({
          url: 'http://192.168.43.31:80/reco',
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
recoG(){
  var that = this
  wx.downloadFile({
    url: 'http://192.168.43.31/camera3',
    header: {
      'content-type': 'application/x-www-form-urlencoded',
    },
    success(res) {

      // filePath2:res.tempFilePath
      // console.log(that.data.filePath2)
      that.setData({
        imageSrc : res.tempFilePath
      });
    }

  })

},

retu() {
    wx.navigateTo({
      url: '../detail/detail',
    })
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
