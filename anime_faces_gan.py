import tensorflow as tf
import os
import matplotlib.pyplot as plt

###################
!ls -lha kaggle.json
!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json

!kaggle datasets download -d splcher/animefacedataset

!unzip animefacedataset
###################
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

tf.config.list_physical_devices()

image=tf.data.Dataset.list_files(os.path.join('images','*jpg')).as_numpy_iterator()

def load_image(path):
    image=tf.io.read_file(path)
    image=tf.io.decode_jpeg(image)
    image=tf.image.resize(image,size=(64,64))
    image=image/255.0
    return image