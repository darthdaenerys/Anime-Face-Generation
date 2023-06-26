import tensorflow as tf
import os
import matplotlib.pyplot as plt

##################
!ls -lha kaggle.json
!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json

!kaggle datasets download -d splcher/animefacedataset

!unzip animefacedataset
##################
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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2DTranspose,Reshape,BatchNormalization,Conv2D,MaxPool2D,Flatten
import numpy as np

def build_generator():
    model=Sequential(name='generator')

    model.add(Dense(4*4*128,input_shape=(128,)))
    model.add(Reshape((4,4,128)))
    model.add(Conv2DTranspose(64,(4,4),2,padding='same',activation='relu',kernel_initializer='random_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64,(4,4),2,padding='same',activation='relu',kernel_initializer='random_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(32,(4,4),2,padding='same',activation='relu',kernel_initializer='random_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(32,(4,4),2,padding='same',activation='relu',kernel_initializer='random_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(3,(4,4),1,padding='same',activation='sigmoid',kernel_initializer='random_uniform'))
    return model

generator=build_generator()
generator.summary()

tf.keras.utils.plot_model(generator,show_shapes=True,show_layer_activations=True,to_file='architecture/generator.png')

images=generator.predict(np.random.randn(128,128),verbose=0)
show_images(images)

def build_discriminator():
    model=Sequential(name='discriminator')

    model.add(Conv2D(32,(5,5),1,input_shape=(64,64,3)))
    model.add(MaxPool2D())
    model.add(Conv2D(64,(5,5),1,activation='relu',kernel_initializer='he_normal'))
    model.add(MaxPool2D())
    model.add(Conv2D(128,(3,3),1,activation='relu',kernel_initializer='he_normal'))
    model.add(Conv2D(512,(3,3),1,activation='relu',kernel_initializer='he_normal'))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))

    return model

discriminator=build_discriminator()
print(discriminator.summary())

tf.keras.utils.plot_model(discriminator,show_shapes=True,show_layer_activations=True,to_file='architecture/discriminator.png')

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

generator_opt=Adam(learning_rate=0.0004)
discriminator_opt=Adam(learning_rate=0.001)
generator_loss=BinaryCrossentropy()
discriminator_loss=BinaryCrossentropy()

class AnimeGAN(tf.keras.models.Model):
    def __init__(self,generator,discriminator,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.generator=generator
        self.discriminator=discriminator
    
    def compile(self,g_opt,g_loss,d_opt,d_loss,*args,**kwargs):
        super().compile(*args,**kwargs)
        self.generator_loss=g_loss
        self.generator_opt=g_opt
        self.discriminator_loss=d_loss
        self.discriminator_opt=d_opt
    
    def train_step(self,batch):
        # fetch data
        fake_images=self.generator(np.random.randn(128,128),training=False)
        real_images=batch

        # train discriminator
        with tf.GradientTape() as d_tape:
            yhat_real=self.discriminator(real_images,training=True)
            yhat_fake=self.discriminator(fake_images,training=True)
            yhat_realfake=tf.concat([yhat_real,yhat_fake],axis=0)
            y_realfake=tf.concat([tf.zeros_like(yhat_real),tf.ones_like(yhat_fake)],axis=0)
            noise_real = 0.15*tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15*tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)
            d_loss = self.discriminator_loss(y_realfake, yhat_realfake)
            
        # apply backpropagation to train discriminator
        d_grad=d_tape.gradient(d_loss,self.discriminator.trainable_variables)
        self.discriminator_opt.apply_gradients(zip(d_grad,self.discriminator.trainable_variables))

        with tf.GradientTape() as g_tape:
            gen_images = self.generator(tf.random.normal((128,128)), training=True)
            predicted_labels = self.discriminator(gen_images, training=False)
            g_loss = self.generator_loss(tf.zeros_like(predicted_labels), predicted_labels) 

        ggrad = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        return {'g_loss':g_loss,'d_loss':d_loss}