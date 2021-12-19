import * from Unet

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

def plot_history(tr_loss, val_loss):
# plot Loss
 pyplot.plot(tr_loss, "-g",  label='Training Loss')
 pyplot.plot(val_loss, "-r",  label='Validation Loss')
 pyplot.gca().set_title("Loss Curve")
 pyplot.legend(loc="upper right")
 pyplot.show()

data_path = '/content/drive/MyDrive/CSE-803-Project/'

image_list = [] #Array to store name of the image.

for i in  range(4):
  folder=i.__str__().zfill(4)
  loc = data_path+'danbooru-sketch-pair-128x/color/sketch/' + folder + '/'
  for filename in os.listdir(loc):    
    image_list.append(folder+'/'+filename)  

image_training  = image_list[0:7000]
image_validation = image_list[7000:7300]
image_test = image_list[7300:7600]

model = color_image()

train_batch_starting_point = 0
validation_batch_starting_point = 0

epochs = 100    #Number of epochs.
batch_size = 128 #Batch Size.

#Length of train dataset.
len_tr = len(image_training) 

#List to store results.
tr_loss_hist, val_loss_hist = list(), list()

#Calling initial model to generate images
model = color_image()

#Timer
start_time = datetime.now()

#Training for number of epochs.
for e in range(epochs):

  #Looping for number of iterations so that all images gets processed atleast once in epoch.
  for i in range(int(len_tr/batch_size)):
  
    #Generate batch of train data.
    xs, ys = get_train_batch(image_training, batch_size)
   
    #Train model on batch of data generated above.   
    tr_loss = model.train_on_batch(xs,ys)

    #Generate batch of validation data.
    xs, ys = get_validation_batch(image_validation, batch_size)

    #Test model on validation data
    val_loss = model.test_on_batch(xs,ys)    

    #Append performances.
    tr_loss_hist.append(tr_loss)
    val_loss_hist.append(val_loss)
  
  #Print results after every 25 epoch.
  if e == 1 or e % 25 == 0 or e% (epochs-1) == 0:
     print('Iteration number:',e)   
     elapsed_time = datetime.now() - start_time
     print('Elapsed Time:', elapsed_time)
     print('Training Loss:',tr_loss)
     print('Validation Loss:',val_loss) 
     if  (e % 25 == 0 or e% (epochs-1) == 0) and (e != 0):
         plot_history(tr_loss_hist, val_loss_hist) 
         #Saving the model.
         model.save_weights('model_' + str(e) + '.h5')
         print("Saved model to disk")
     #Generating images using model trained.    
     summarize_performance(model)

     