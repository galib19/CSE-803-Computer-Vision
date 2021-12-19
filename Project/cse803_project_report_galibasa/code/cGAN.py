

def create_generator():

    
    
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
    

    
    
    
    
    
    
    
    encoder_output_6 = Conv2DTranspose(512, (5,5), padding='same', activation='relu', strides=2, kernel_initializer=init )(encoder_output_5)
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
    
    encoder_output = Conv2DTranspose(3, (5,5), padding='same', activation='tanh', strides=2, kernel_initializer=init, name="gen_output")(encoder_output_9)
    
    
    model = Model(encoder_input, encoder_output)
    
    return model
def create_discriminator():

    
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
    
    encoder_output_5 = Flatten()(encoder_output_5)
    
    encoder_output = Dense(units=1, activation='sigmoid', name="disc_output")(encoder_output_5)
    
    model = Model(encoder_input, encoder_output)   
    
    opt = Adam(lr=0.0002, beta_1=0.5) 
    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

    def combine_gan(discriminator, generator): 
    
    discriminator.trainable=False

    
    gan_input = Input(shape=(128,128,3))

    
    x = generator(gan_input)

    
    col_input = Input(shape=(128,128,3))

    
    pixLevelLoss = L1_loss(col_input, x)

    
    gan_output= discriminator(x)

    
    gan= Model(inputs=[gan_input, col_input], outputs=gan_output)

    
    opt = Adam(lr=0.0002, beta_1=0.5)  
    
    
    gan.compile(optimizer=opt, loss=lambda y_true, y_pred : losses.binary_crossentropy(y_true, y_pred) + \
               100 *  pixLevelLoss(y_true, y_pred), metrics=['accuracy'])
    
    
    return gan

def get_train_batch(arr, n_samples, generator):




  
  
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

  
  X_res = generator.predict(X_sketches)

  
  return X_sketches, X_color, X_res

  def get_validation_batch(arr, n_samples, generator):




  
  
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


  
  X_res = generator.predict(X_sketches)

  
  return X_sketches, X_color, X_res

def summarize_performance(model):

  
  gen_arr = model.predict(X_train)

  
  for i in range(3):
    print('Train data : sketch image')
    
    img = X_train[i] *255.0
    
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR)
    
    cv2_imshow(img)
    print('Train data : color image')
    
    img = ((Y_train[i]+ 1)/2.0)*255.0
    
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR)
    
    cv2_imshow(img)
    print('Train data : Generated image')
    
    img = ((gen_arr[i]+ 1)/2.0)*255.0
    
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR)
    
    cv2_imshow(img)
  
  
  gen_arr = model.predict(X_validation)

  
  for i in range(3):
    print('Validation data : sketch image')
    
    img = ((X_validation[i]+ 1)/2.0)*255.0
    
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR)
    
    cv2_imshow(img)
    print('Validation data : color image')
    
    img = ((Y_validation[i]+ 1)/2.0)*255.0
    
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR)
    
    cv2_imshow(img)
    print('Validation data : Generated image')
    
    img = ((gen_arr[i]+ 1)/2.0)*255.0
    
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR)
    
    cv2_imshow(img)

    def training_cgan(epochs=1, batch_size=128):

    
    start_time = datetime.now()

    
    global train_batch_starting_point
    
    
    generator= create_generator()

    
    discriminator= create_discriminator()
    
    
    gan = combine_gan(discriminator, generator)

    
    d_loss_hist, g_loss_hist, d_acc_hist, g_acc_hist = list(), list(), list(), list()
    g_loss_hist_val,  g_acc_hist_val = list(), list()
    
    for e in range(epochs):
        
        
        half_batch = int(batch_size / 2)

        
        if not e%2:
           
           skc, col, res =  get_train_batch(image_training, half_batch, generator) 
           
           
           train_batch_starting_point -= half_batch

           
           y_real = np.ones((half_batch, 1))

           
           y_fake = np.zeros((half_batch, 1))

           
           discriminator.trainable=True

           
           d_loss_real, d_acc_real = discriminator.train_on_batch(col, y_real * .9)
        
           
           d_loss_fake, d_acc_fake = discriminator.train_on_batch(res, y_fake)

           
           d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

           
           d_acc = 0.5 * np.add(d_acc_real, d_acc_fake)

           
           d_loss_hist.append(d_loss)
           d_acc_hist.append(d_acc)

        
        skc, col, res = get_train_batch(image_training, batch_size, generator)  

        
        y_real = np.ones((batch_size, 1))
   
        
        g_loss, g_acc = gan.train_on_batch([skc,col], y_real)
        

        
        skc, col, res = get_validation_batch(image_validation, batch_size, generator)  
   
        
        g_loss_val, g_acc_val = gan.test_on_batch([skc,col], y_real)
        
        
        g_loss_hist.append(g_loss)  
        g_acc_hist.append(g_acc)    
        g_loss_hist_val.append(g_loss_val)  
        g_acc_hist_val.append(g_acc_val)    

        
        if e == 1 or e % 15000 == 0 or e% (epochs-1) == 0:
           print('Iteration number:',e)   
           elapsed_time = datetime.now() - start_time
           print('Elapsed Time:', elapsed_time)
           print('Generator Loss Train:',g_loss)
           print('Generator Loss Val:',g_loss_val)
           print('Generator Accuracy Train:',g_acc)
           print('Generator Accuracy Val:',g_acc_val) 
           print('Discriminator Loss:',d_loss)
           print('Discriminator Accuracy:',d_acc)       

           
           if  (e % 15000 == 0 or e% (epochs-1) == 0) and (e != 0):
               
               plot_history(d_loss_hist, g_loss_hist, d_acc_hist, g_acc_hist, g_loss_hist_val, g_acc_hist_val) 
           
           summarize_performance(generator)
           
           generator.save_weights('model_gen_' + str(e) + '.h5')
           discriminator.save_weights('model_dis_' + str(e) + '.h5')
           print("Saved model to disk")