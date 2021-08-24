import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os 
import pathlib
import random
import matplotlib.pyplot as plt



os.environ['TF_CPP_MIN_LOG_LEVEL']='2'



def read_data(path):
    path_root = pathlib.Path(path)
    # print(path_root)
    # for item in path_root.iterdir():
    #     print(item)
    image_paths = list(path_root.glob('*/*'))
    image_paths = [str(path) for path in image_paths]
    random.shuffle(image_paths)
    image_count = len(image_paths)
    # print(image_count)
    # print(image_paths[:10])

    label_names = sorted(item.name for item in path_root.glob('*/') if item.is_dir())
    # print(label_names)
    label_name_index = dict((name, index) for index, name in enumerate(label_names))
    # print(label_name_index)
    image_labels = [label_name_index[pathlib.Path(path).parent.name] for path in image_paths]
    # print("First 10 labels indices: ", image_labels[:10])
    return image_paths,image_labels,image_count


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [100, 100])
    image /= 255.0  # normalize to [0,1] range
    # image = tf.reshape(image,[100*100*3])
    return image

def load_and_preprocess_image(path,label):
    image = tf.io.read_file(path)
    return preprocess_image(image),label

def creat_dataset(image_paths,image_labels,bitch_size):
    db = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
    dataset = db.map(load_and_preprocess_image).batch(bitch_size)    
    return dataset


def train_model(train_data,test_data):
    #构建模型
    network = keras.Sequential([
            keras.layers.Conv2D(32,kernel_size=[5,5],padding="same",activation=tf.nn.relu),
            keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
            keras.layers.Conv2D(64,kernel_size=[3,3],padding="same",activation=tf.nn.relu),
            keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
            keras.layers.Conv2D(64,kernel_size=[3,3],padding="same",activation=tf.nn.relu),
            keras.layers.Flatten(),
            keras.layers.Dense(512,activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128,activation='relu'),
            keras.layers.Dense(10)])
    network.build(input_shape=(None,100, 100,3))
    #(100, 100)
    network.summary()

    network.compile(optimizer=optimizers.SGD(lr=0.001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
    )
    #模型训练
    network.fit(train_data, epochs = 10,validation_data=test_data,validation_freq=2)  
    network.evaluate(test_data)

    tf.saved_model.save(network,'D:\\code\\PYTHON\\gesture_recognition\\model\\')
    print("保存模型成功")



    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: network(x))
    full_model = full_model.get_concrete_function(
    tf.TensorSpec(network.inputs[0].shape, network.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
            logdir="D:\\code\\PYTHON\\gesture_recognition\\model\\frozen_model\\",
            name="frozen_graph.pb",
            as_text=False)
    print("模型转换完成，训练结束")


if  __name__ == "__main__":
    print(tf.__version__)
    train_path = 'D:\\code\\PYTHON\\gesture_recognition\\Dataset'
    test_path = 'D:\\code\\PYTHON\\gesture_recognition\\testdata' 
    image_paths,image_labels,_ = read_data(train_path)
    train_data = creat_dataset(image_paths,image_labels,16)
    image_paths,image_labels,_ = read_data(test_path)
    test_data = creat_dataset(image_paths,image_labels,16)
    train_model(train_data,test_data)
    
