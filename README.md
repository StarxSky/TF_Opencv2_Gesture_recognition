# TF_Opencv2_Gesture_recognition
一个基于tensorflow2.6.0和OpenCV2的手势识别(自制数据集)
[Download Code](https://github.com/Xhs753/TF_Opencv2_Gesture_recognition/archive/refs/heads/main.zip)


![image](https://github.com/Xhs753/TF_Opencv2_Gesture_recognition/blob/main/IMG_20210824_185701.jpg?raw=true)
![](003.png)

## 数据集

-- 有关于这个数据集目前我只是搞了一小部分，过后我将会不断添加。
-- 数据集的每张图片都已经标好序号请切勿打乱
-- 数据集的图片采用的是150 x 150的分辨率
-- 声明：图片以及数据集的代码均为本人原创
### 食用方法
1. 训练开始前请务必将代码中的文件路径放正确！！
2. train.py和CV.py在Main文件夹中
3. 使用python运行train.py进行训练模型
4. 使用python运行CV.py以打开实时识别
5. 如果结果不准确可以考虑代码中的epochs的次数(建议：100)

### 安装TF
 ```
 pip install tensorflow 
 ```
### Install OpenCV
``` 
pip install opencv-python==4.1.2
```
#### 路径导致的错误解决方案
如果在您的计算机上有D盘请务必按照代码中的文件路径创建！！（盘名可以随意更改！但是文件路径一定要写正确！！
 训练图片的入口：
 ```
 train_dir = 'D:\\code\\PYTHON\\gesture_recognition\\Dataset'
 ```
 
 #### 代码中的保存权重的文件请按照如下例子制作注意！！一定要预先创建gestureModel_one.h5文件
 ```
 network.save_weights('D:\\code\\model_save\\gesture_recognition_model\\gestureModel_one.h5')
 ```
 #### 保存模型
 ```
 tf.saved_model.save(network, 'D:\\code\\model_save\\gesture_recognition_model\\gestureModel_one')
 ```
##### 新！！
 1. 本次将所有功能集结到同一个train.py中！！
 ---更新2021/8/26
 2. 本次更改了图片属性（分辨率提升为150 x 150)
 ---更新2021/8/25
  

##### 训练截图
![](https://github.com/Xhs753/TF_Opencv2_Gesture_recognition/blob/main/-1fb6f7631e238c27.png?raw=true)
![](https://github.com/Xhs753/TF_Opencv2_Gesture_recognition/blob/main/-760d7da8022e2d0.png?raw=true)


###### 鸣谢
感谢各位大佬们的支持，本人是个高中生在写代码时难免会有些错误还望大家多多支持并在issues里发表意见
我会认真改进！！
也希望给个Star![10C0F9CD7ADFCAD62214B11420F25D24](https://user-images.githubusercontent.com/62407841/130604673-0fb083df-b7bc-4d67-9742-72cc223dcc1e.png)
支持一下呗~ [/比心]
