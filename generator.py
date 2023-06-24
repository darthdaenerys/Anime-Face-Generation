import os,numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from tensorflow.keras.models import load_model
from matplotlib.animation import FuncAnimation

generator=load_model(os.path.join('models','generator.h5'))
def show_images(i):
    idx=0
    images=generator.predict(np.random.randn(32,128),verbose=0)
    for row in range(4):
        for col in range(8):
            ax[row][col].imshow(images[idx],animated=True)
            ax[row][col].axis('off')
            idx+=1

fig,ax=plt.subplots(nrows=4,ncols=8,figsize=(40,20))
fig.tight_layout(pad=0,h_pad=0,w_pad=0)
anim = FuncAnimation(fig, show_images,frames=30, interval=2000)
plt.show()