import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os
import pathlib
import random
import matplotlib.pyplot as plt


data_root = pathlib.Path('D:\code\PYTHON\gesture_recognition\Dataset')
print(data_root)
for item in data_root.iterdir():
  print(item)


all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
image_count = len(all_image_paths)
print(image_count)    ##统计共有多少图片
for i in range(10):
  print(all_image_paths[i])


label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print(label_names) #标签名其实就是文件夹的名字
label_to_index = dict((name, index) for index, name in enumerate(label_names))
print(label_to_index)
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

print("First 10 labels indices: ", all_image_labels[:10])


def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [100, 100])
  image /= 255.0  # normalize to [0,1] range
  # image = tf.reshape(image,[100*100*3])
  return image

def load_and_preprocess_image(path,label):
  image = tf.io.read_file(path)
  return preprocess_image(image),label


ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

train_data = ds.map(load_and_preprocess_image).batch(6)#16

