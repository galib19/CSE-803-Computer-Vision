import * from cGAN 

from keras.initializers import RandomNormal
init = RandomNormal(stddev=0.02)

from keras.layers import Activation, Dense, Dropout, Flatten, Input, Add, Conv2D, Conv2DTranspose, LeakyReLU #Keras Layers.
from tensorflow.keras.layers import (
    BatchNormalization, Dropout, Dense
)
from keras.models import Model 
from tensorflow.keras.optimizers import Adam.
from keras import losses 
import numpy as np 
import os 
import random 
import cv2 
from google.colab.patches import cv2_imshow 
import matplotlib.pyplot as pyplot 
from datetime import datetime 

import tensorflow.keras.backend as K

def L1_loss(y, g):

  def finalPLLoss(y_true, y_pred):
    return K.mean( K.abs( y - g ) )

  return finalPLLoss

def plot_history(d_loss_hist, g_loss_hist, d_acc_hist, g_acc_hist, g_loss_hist_val,  g_acc_hist_val):

 pyplot.subplot(1, 1, 1)
 pyplot.plot(d_loss_hist,'-r', label='Discriminator Loss')
 pyplot.gca().set_title("Discriminator Loss")
 pyplot.legend()
 pyplot.show()
 pyplot.subplot(1, 1, 1)
 pyplot.plot(g_loss_hist, '-g', label='Generator Loss (Train)')
 pyplot.plot(g_loss_hist_val, '-b', label='Generator Loss (Validation)')
 pyplot.gca().set_title("Generator Loss")
 pyplot.legend()
 pyplot.show()


data_path = '/content/drive/MyDrive/CSE-803-Project/'

image_list = [] 

for i in  range(4):
  folder=i.__str__().zfill(4)
  loc = data_path+'danbooru-sketch-pair-128x/color/sketch/' + folder + '/'
  for filename in os.listdir(loc):    
    image_list.append(folder+'/'+filename)  

image_training  = image_list[0:7000]
image_validation = image_list[7000:7300]
image_test = image_list[7300:7600]


generator = create_generator()

d =create_discriminator()

gan = combine_gan(create_discriminator(),create_generator())

n_samples = 

train_batch_starting_point = 0
validation_batch_starting_point = 0

X_train, Y_train, genated_images_train = get_train_batch(image_training, n_samples, generator)
X_validation, Y_validation, generated_images_validation = get_validation_batch(image_validation, n_samples, generator)

train_batch_starting_point = 0
validation_batch_starting_point = 0

training_cgan(62500,64)