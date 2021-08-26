import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os
import pathlib
import random
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_roi(frame, x1, x2, y1, y2):
    dst = frame[y1:y2, x1:x2]
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
    return dst

def get_image(image, network):

    # image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.expand_dims(image, axis=2)  # 扩维度,让单通道图片可以resize
    print("image.shape =  ", image.shape)
    image = tf.image.resize(image, [100, 100])
    # image = cv.resize(image, (100, 100))
    # image = image.reshape(-1, 100, 100,  1)
    image1 = image / 255.0  # normalize to [0,1] range
    image1 = tf.expand_dims(image1, axis=0)
    # print(image1.shape)
    pred = network(image1)
    print("预测结果原始结果", pred)
    print()
    #pred是个字典，里面没有'output_1',pred = tf.nn.softmax(pred['dense_2'], axis=1)
    pred = tf.nn.softmax(pred['dense_2'], axis=1)
    print("预测softmax后", pred)
    pred = tf.argmax(pred, axis=1)
    print("最终测试结果", pred.numpy())
    cv.putText(frame, "Predicted results = :" + str(pred.numpy()), (100, 400), cv.FONT_HERSHEY_SIMPLEX,
               1, [0, 255, 255])

if __name__ == "__main__":

    capture = cv.VideoCapture(0)
    #creatTrackbar()
    channels = 3
    DEFAULT_FUNCTION_KEY = "serving_default"
    loaded = tf.saved_model.load('D:\\code\\model_save\\gesture_recognition_model\\gestureModel_one\\')
    network = loaded.signatures[DEFAULT_FUNCTION_KEY]
    print(list(loaded.signatures.keys()))
    print('加载 weights 成功')
    while True :

        ret, frame = capture.read()
        roi = get_roi(frame, 100, 250, 100, 250)
        cv.imshow("roi", roi)
        get_image(roi, network)
        cv.imshow("frame", frame)
        c = cv.waitKey(50)
        if c == 27:
            break
    cv.waitKey(0)
    capture.release()
    cv.destroyAllWindows()

