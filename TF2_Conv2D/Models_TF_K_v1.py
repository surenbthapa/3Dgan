#file which contains neural network models and loss functions

#v3: no combined model and bias

import tensorflow as tf
import numpy as np
import time
import sys

def generator(latent_size=200, keras_dformat='channels_first'):
 
    dim = (5,5,5)
    latent = tf.keras.Input(shape=(latent_size ))  #define Input     
    x = tf.keras.layers.Dense(5*5*5, input_dim=latent_size)(latent)   #shape (none, 625) #none is batch size
    x = tf.keras.layers.Reshape(dim) (x)       #shape after (none, 5,5,5)  
    
    x1 = x
    x2 = tf.keras.layers.Permute([3,1,2])(x)   #permute starts indexing with 1, same as tf.transpose
    x3 = tf.keras.layers.Permute([2,3,1])(x)   
    
    def path(x):
        #path1
        #1.Conv Block
        x = tf.keras.layers.Conv2D(32, (5, 5), data_format=keras_dformat, use_bias=False, padding='same')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization() (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        #2.Conv Block
        x = tf.keras.layers.Conv2DTranspose(64, (5,5), strides =(3,3), data_format=keras_dformat, padding="same") (x)
        x = tf.keras.layers.Conv2D(64, (5, 5), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization() (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        #3.Conv Block
        x = tf.keras.layers.Conv2DTranspose(64, (5,5), strides =(2,2), data_format=keras_dformat, padding="same") (x)
        x = tf.keras.layers.Conv2D(64, (8, 8), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization() (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        #4. Conv Block
        x = tf.keras.layers.Conv2D(64, (5, 5), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization() (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        #5. Conv Block
        x = tf.keras.layers.Conv2D(32, (4, 4), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization() (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        #6. Conv Block
        x = tf.keras.layers.Conv2D(32, (3, 3), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization() (x)  
        x = tf.keras.layers.Dropout(0.2)(x)
        #7.Conv Block
        x = tf.keras.layers.Conv2D(25, (2, 2), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization() (x)  
        x = tf.keras.layers.Dropout(0.2)(x)
        return x

    x1 = path(x1)
    x2 = path(x2)
    x3 = path(x3)

    x2 = tf.keras.layers.Permute([2,3,1])(x2)   #permute starts indexing with 1
    x3 = tf.keras.layers.Permute([3,1,2])(x3)   #permute starts indexing with 1

    x = tf.keras.layers.concatenate([x1,x2,x3],axis=1) #i stack them on the channels axis
 
    x = tf.keras.layers.Conv2D(25, (3,3), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Reshape([1,25,25,25])(x)
    return tf.keras.Model(inputs=[latent], outputs=x)   


def discriminator(keras_dformat='channels_first'):
   # print ("keras_dformat: ", keras_dformat)
    if keras_dformat =='channels_last':
        dshape=(25, 25, 25,1)
        daxis=(1,2,3)
    else:
        dshape=(1, 25, 25, 25)
        daxis=(2,3,4)
        
    image = tf.keras.layers.Input(shape=dshape)     #Input Image
    x = image
    x = tf.keras.layers.Reshape([25,25,25])(x)
    
    x1 = x
    x2 = tf.keras.layers.Permute([3,1,2])(x)   
    x3 = tf.keras.layers.Permute([2,3,1])(x)   
    
    def path(x):
        #path1
        #1.Conv Block
        x = tf.keras.layers.Conv2D(64, (8, 8), data_format=keras_dformat, use_bias=False, padding='same')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization() (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        #2.Conv Block
        x = tf.keras.layers.Conv2D(32, (6, 6), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization() (x)
        x = tf.keras.layers.Dropout(0.2)(x)
        #3.Conv Block
        x = tf.keras.layers.Conv2D(32, (5, 5), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization() (x)
        x = tf.keras.layers.Dropout(0.2)(x)
        #4. Conv Block
        x = tf.keras.layers.Conv2D(32, (4, 4), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization() (x)
        x = tf.keras.layers.Dropout(0.2)(x)
        #5. Conv Block
        x = tf.keras.layers.Conv2D(32, (6, 6), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization() (x)
        x = tf.keras.layers.Dropout(0.2)(x)
        #6. Conv Block
        x = tf.keras.layers.Conv2D(32, (3, 3), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization() (x)
        x = tf.keras.layers.Dropout(0.2)(x)
        #7. Conv Block
        x = tf.keras.layers.Conv2D(9, (3, 3), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization() (x)
        x = tf.keras.layers.Dropout(0.2)(x)
        return x

    x1 = path(x1)
    x2 = path(x2)
    x3 = path(x3)
    
    x2 = tf.keras.layers.Permute([2,3,1])(x2)   #permute starts indexing with 1
    x3 = tf.keras.layers.Permute([3,1,2])(x3)   #permute starts indexing with 1
    
    x = tf.keras.layers.concatenate([x1,x2,x3],axis=1) #i stack them on the channels axis

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(10, activation='linear')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    fake = tf.keras.layers.Dense(1, activation='sigmoid', name='generation')(x)   #true/fake classification
    aux = tf.keras.layers.Dense(1, activation='linear', name='auxiliary')(x)      #auxiliary energy 
    ecal = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis=daxis))(image)    #sum of pixels of imput image

    return tf.keras.Model(inputs=image, outputs=[fake, aux, ecal])



def bit_flip_tf(x, prob = 0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1* np.logical_not(x[selection])
    x = tf.constant(x)
    return x


def func_for_gen_tf(nb_test, latent_size = 200):   #generate input data for generator
    noise =            tf.Variable(tf.random.normal([nb_test,latent_size], mean=0.0, stddev = 1.0)) #input for bit_flip() to generate true/false values for discriminator
    gen_aux =          tf.Variable(tf.random.uniform([nb_test,1 ],1, 5,))   #generates aux for dicriminator
    gen_ecal =         tf.math.multiply(2, gen_aux)                          #generates ecal for discriminator
    generator_input =  tf.math.multiply(gen_aux, noise)                      #generates input for generator
    return noise, gen_aux, generator_input, gen_ecal


def disc_loss(generator, discriminator,image_batch, energy_batch, ecal_batch, batch_size, label, latent_size=200):
    discriminate = discriminator(image_batch)

    #true/fake loss, cross entropy
    if label == "ones":
        labels = bit_flip_tf(tf.ones_like(discriminate[0])*0.9)     #true=1    
    elif label == "zeros":
        labels = bit_flip_tf(tf.zeros_like(discriminate[0])*0.1)    #fake=0
    loss_true_fake = tf.reduce_mean(- labels * tf.math.log(discriminate[0] + 2e-7) - (1 - labels) * tf.math.log(1 - discriminate[0] + 2e-7))

    #aux loss, MAPE
    loss_aux = tf.reduce_mean(tf.math.abs((energy_batch - discriminate[1])/(energy_batch + 2e-7))) *100
        
    #ecal loss
    loss_ecal = tf.reduce_mean(tf.math.abs((ecal_batch - discriminate[2])/(ecal_batch + 2e-7))) *100
   
    #total loss
    weight_true_fake = 6.0
    weight_aux = 0.2
    weight_ecal = 0.1
    total_loss = weight_true_fake * loss_true_fake + weight_aux * loss_aux + weight_ecal * loss_ecal
    return total_loss, loss_true_fake, loss_aux, loss_ecal


def gen_loss(generator, discriminator, batch_size, latent_size=200):
    noise, gen_aux, generator_input, gen_ecal = func_for_gen_tf(nb_test=batch_size, latent_size=latent_size) 
    generated_images = generator(generator_input)
    discriminator_fake = discriminator(generated_images)
    
    #true/fake
    label_fake = bit_flip_tf(tf.ones_like(discriminator_fake[0])*0.9)   #ones = true
    loss_true_fake = tf.reduce_mean(- label_fake * tf.math.log(discriminator_fake[0] + 2e-7) - 
                               (1 - label_fake) * tf.math.log(1 - discriminator_fake[0] + 2e-7))
    
    #aux
    loss_aux = tf.reduce_mean(tf.math.abs((gen_aux - discriminator_fake[1])/(gen_aux + 2e-7))) *100
    
    #ecal
    loss_ecal = tf.reduce_mean(tf.math.abs((gen_ecal - discriminator_fake[2])/(gen_ecal + 2e-7))) *100
    
    #total loss
    weight_true_fake = 6.0
    weight_aux = 0.2
    weight_ecal = 0.1
    total_loss = weight_true_fake * loss_true_fake + weight_aux * loss_aux + weight_ecal * loss_ecal
    return total_loss, loss_true_fake, loss_aux, loss_ecal












