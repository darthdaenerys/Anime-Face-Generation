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

img=load_image(image.next())
print(img.shape)
plt.imshow(img)
plt.tight_layout()
plt.axis("off")
plt.show()

data=tf.data.Dataset.list_files(os.path.join('images','*jpg'))
data=data.map(load_image)
data=data.shuffle(10000)
data=data.batch(256)
data=data.prefetch(tf.data.AUTOTUNE)
data_iterator=data.as_numpy_iterator()

def show_images(images):
    idx=0
    fig,ax=plt.subplots(nrows=4,ncols=8,figsize=(20,10))
    for row in range(4):
        for col in range(8):
            ax[row][col].imshow(images[idx])
            ax[row][col].axis('off')
            idx+=1
    fig.tight_layout()
  
def get_figure():
    images=generator.predict(np.random.randn(128,128),verbose=0)
    idx=0
    fig,ax=plt.subplots(nrows=4,ncols=8,figsize=(20,10))
    for row in range(4):
        for col in range(8):
            ax[row][col].imshow(images[idx])
            ax[row][col].axis('off')
            idx+=1
    fig.tight_layout()
    return fig

images=data_iterator.next()
show_images(images)