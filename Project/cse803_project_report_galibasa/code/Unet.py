

def get_train_batch(arr, n_samples):




  
  
  global train_batch_starting_point

  
  X_sketches = []
  X_color = []

  
  for i in range(n_samples):
    
    
    loc = data_path+'danbooru-sketch-pair-128x/color/sketch/'
    
    img = cv2.imread(loc+arr[(i + train_batch_starting_point) % len(arr)],1)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    X_sketches.append(img)

    
    loc = data_path+'danbooru-sketch-pair-128x/color/src/' 
    
    img = cv2.imread(loc+arr[(i + train_batch_starting_point) % len(arr)],1)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    X_color.append(img)

  
  train_batch_starting_point +=  n_samples
  
  
  X_sketches = np.array(X_sketches)
  X_color = np.array(X_color)

  
  X_sketches = (X_sketches - 127.5) / 127.5
  X_color = (X_color - 127.5) / 127.5


  
  return X_sketches, X_color
  
def get_validation_batch(arr, n_samples):




  
  
  global validation_batch_starting_point

  
  X_sketches = []
  X_color = []

  
  for i in range(n_samples):
    
    loc = data_path+'danbooru-sketch-pair-128x/color/sketch/'
    
    img = cv2.imread(loc+arr[(i + validation_batch_starting_point) % len(arr)],1)
    
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    X_sketches.append(img)

    
    loc = data_path+'danbooru-sketch-pair-128x/color/src/' 
    
    img = cv2.imread(loc+arr[(i + validation_batch_starting_point) % len(arr)],1)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    X_color.append(img)

  
  validation_batch_starting_point +=  n_samples
  
  
  X_sketches = np.array(X_sketches)
  X_color = np.array(X_color)

  
  X_sketches = (X_sketches - 127.5) / 127.5
  X_color = (X_color - 127.5) / 127.5


  
  return X_sketches, X_color

  
  def color_image():

    
    
    encoder_input = Input(shape=(128, 128, 3,)) 
    
    encoder_output_1 = Conv2D(64, (5,5), padding='same',  strides=2, kernel_initializer=init)(encoder_input) 
    
    encoder_output_2 = LeakyReLU(alpha=0.2)(encoder_output_1)  
    
    encoder_output_2 = Conv2D(128, (5,5),  padding='same', strides=2, kernel_initializer=init)(encoder_output_2)
    
    encoder_output_2 = BatchNormalization(momentum=0.9)(encoder_output_2)
    
    encoder_output_3 = LeakyReLU(alpha=0.2)(encoder_output_2)
    encoder_output_3 = Conv2D(256, (5,5),  padding='same',  strides=2, kernel_initializer=init)(encoder_output_3)
    encoder_output_3 = BatchNormalization(momentum=0.9)(encoder_output_3)
    
    encoder_output_4 = LeakyReLU(alpha=0.2)(encoder_output_3)
    encoder_output_4 = Conv2D(512, (5,5),  padding='same',  strides=2, kernel_initializer=init)(encoder_output_4)
    encoder_output_4 = BatchNormalization(momentum=0.9)(encoder_output_4)
    
    encoder_output_5 = LeakyReLU(alpha=0.2)(encoder_output_4)
    encoder_output_5 = Conv2D(512, (5,5),  padding='same',  strides=2, kernel_initializer=init)(encoder_output_5)
    encoder_output_5 = BatchNormalization(momentum=0.9)(encoder_output_5)
    

    
    
    
    
    
    
    
    encoder_output_6 = Conv2DTranspose(512, (5,5), padding='same', activation='relu', strides=2, kernel_initializer=init)(encoder_output_5)
    encoder_output_6  = BatchNormalization(momentum=0.9)(encoder_output_6)
    
    encoder_output_6 = Add()([encoder_output_6, encoder_output_4])
    
    encoder_output_7 = Conv2DTranspose(256, (5,5), padding='same', activation='relu', strides=2, kernel_initializer=init)(encoder_output_6)
    encoder_output_7  = BatchNormalization(momentum=0.9)(encoder_output_7)
    
    encoder_output_7 = Add()([encoder_output_7, encoder_output_3])
    
    encoder_output_8 = Conv2DTranspose(128, (5,5), padding='same', activation='relu', strides=2, kernel_initializer=init)(encoder_output_7)
    encoder_output_8  = BatchNormalization(momentum=0.9)(encoder_output_8)
    
    encoder_output_8 = Add()([encoder_output_8, encoder_output_2]) 
    
    encoder_output_9 = Conv2DTranspose(64, (5,5), padding='same', activation='relu', strides=2, kernel_initializer=init)(encoder_output_8)
    encoder_output_9  = BatchNormalization(momentum=0.9)(encoder_output_9)
    
    encoder_output_9 = Add()([encoder_output_9, encoder_output_1]) 
    
    encoder_output = Conv2DTranspose(3, (5,5), padding='same', activation='tanh', strides=2, kernel_initializer=init)(encoder_output_9)
    
    
    model = Model(encoder_input, encoder_output)

    
    opt = Adam(lr=0.0002, beta_1=0.5)  
    
    
    model.compile(optimizer=opt, loss=['mae'])
    
    return model

