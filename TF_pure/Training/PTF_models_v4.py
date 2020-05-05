#file which contains neural network models and loss functions

#v3: no combined model and bias

import tensorflow as tf
import numpy as np
import time

def dense(x, weights, dropout = 0.0):
    x = tf.matmul(x, weights)    
    x = tf.nn.dropout(x, dropout)
    return x

def conv3d_transpose(x, weights, stride):
    shape = x.get_shape().as_list()
    W = weights #d,h,w,c_out, c_in
    out_shape = [shape[0], weights.shape[3], shape[2]*stride, shape[3]*stride, shape[4]*stride] # n,c_out, d,h,w
    x = tf.nn.conv3d_transpose(input=x, filters=W, output_shape=out_shape,strides= stride, data_format="NCDHW")
    x = tf.nn.leaky_relu(x)  #I dont have the activation in the keras model
    return x

def batch_norm(x, off, scale): #a conv layer with 16 channels has 32 learnable parameters and 32 nonlearnable
    #off = beta, zero initialization;    scale = gamma, one initialization
    mean, variance = tf.nn.moments(x, axes=(0, 2, 3, 4), shift=False, keepdims=True, name=None)
    #mean = moving mean, moving average ;   variance = moving variance
    x = tf.nn.batch_normalization(x, mean, variance, offset=off, scale=scale, variance_epsilon=0.000001, name=None)#the more sparse the images are, the higher can the variance_epsilon value be
    return x

def padding(x, x_pad=1, y_pad=1, z_pad=1):
    shape = x.get_shape().as_list()
    paddings = tf.constant([[0,0],[0,0],[x_pad, x_pad,], [y_pad, y_pad], [z_pad, z_pad]])
    x = tf.pad(x, paddings, "CONSTANT")  #constant values means zeros
    return x

def conv3d(x, w, padding="SAME", dropout=0.0):
    x = tf.nn.conv3d(x, w, padding = padding, strides=[1, 1, 1, 1, 1], data_format="NCDHW")
    x = tf.nn.leaky_relu(x)
    x = tf.nn.dropout(x, dropout)
    return x

def conv3d_no_act(x, w, padding="SAME", dropout=0.0):
    x = tf.nn.conv3d(x, w, padding = padding, strides=[1, 1, 1, 1, 1], data_format="NCDHW")
  #  x = tf.nn.leaky_relu(x)
    x = tf.nn.dropout(x, dropout)
    return x

#@tf.function
def generator(inp, g_weights):
   # x = tf.reshape(input, shape=[-1,200])
    x = inp
    x = dense(x, g_weights[0])
    x = tf.nn.bias_add(x, g_weights[1])#, data_format="NCDHW")
    x = tf.reshape(x, shape=[-1,7,7,7,7])    #num, ch, x,w,h
    x = conv3d(x, g_weights[2], padding="SAME") 
    x = batch_norm(x,g_weights[3],g_weights[4])
    x = conv3d_transpose(x, g_weights[5], stride=4)
    x = tf.nn.bias_add(x, g_weights[6], data_format="NCDHW")
    x = conv3d(x, g_weights[7], padding="SAME") 
    x = batch_norm(x,g_weights[8],g_weights[9])
    x = conv3d(x, g_weights[10], padding="VALID") 
    x = batch_norm(x,g_weights[11],g_weights[12])
    x = conv3d(x, g_weights[13], padding="SAME") 
    x = batch_norm(x,g_weights[14],g_weights[15])
    x = conv3d_no_act(x, g_weights[16], padding="SAME") 
    return x


#build discriminator network
#@tf.function
def discriminator(inp, d_weights):
    x = inp
    x = conv3d(x, d_weights[0], dropout=0.2)
    x = tf.nn.bias_add(x, d_weights[1], data_format="NCDHW")
    x = padding(x,2,2,2)
    x = conv3d(x, d_weights[2], padding="VALID", dropout=0.2)
    x = batch_norm(x,d_weights[3],d_weights[4])
    x = padding(x,2,2,2)
    x = conv3d(x, d_weights[5], padding="VALID", dropout=0.2)
    x = batch_norm(x,d_weights[6],d_weights[7])
    x = padding(x,1,1,1)
    x = conv3d(x, d_weights[8], padding="VALID", dropout=0.2)
    x = batch_norm(x,d_weights[9],d_weights[10])
    x = tf.nn.avg_pool3d(x, ksize=2, strides=2, padding="VALID", data_format='NCDHW')
    x = tf.reshape(x, shape=[tf.shape(x)[0], -1])  # flatten
    #x = tf.compat.v1.layers.flatten( x, name=None, data_format='channels_first')
    #print("flatten: ", x.shape)

    fake = dense(x, d_weights[11])
    fake = tf.nn.sigmoid(fake)
#    print("\nfake: ", fake)
    
    aux = dense(x, d_weights[12])   #this is the identity function (slope one)
#    print("aux: ", aux.shape)
    
    ecal = tf.math.reduce_sum(inp, axis=(2,3,4))
    #print("ecal: ", ecal)
    
    return fake, aux, ecal


def bit_flip_tf(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1* np.logical_not(x[selection])
    x = tf.constant(x)
    return x


def func_for_gen_tf(nb_test, latent_size):
    noise =            tf.Variable(tf.random.normal([nb_test,latent_size], mean=0.0, stddev=1.0)) #input for bit_flip() to generate true/false values for discriminator
    gen_aux =          tf.Variable(tf.random.uniform([nb_test,1 ],1, 5,))   #generates aux for dicriminator
    gen_ecal =         tf.math.multiply(2, gen_aux)                          #generates ecal for discriminator
    generator_input =  tf.math.multiply(gen_aux, noise)                      #generates input for generator
    return noise, gen_aux, generator_input, gen_ecal



def disc_loss(discriminator_true, energy_batch, ecal_batch, g_weights, d_weights, batch_size, latent_size=200):  #energy = aux
    noise, gen_aux, generator_input, gen_ecal = func_for_gen_tf(nb_test=batch_size, latent_size=latent_size) #len has to be batch size, because of bit_flip used below
    generated_images = generator(generator_input, g_weights)
    discriminator_fake = discriminator(generated_images, d_weights)

    #true/fake loss
    label_true = bit_flip_tf(tf.ones_like(discriminator_true[0]))     #true=1    
    label_fake = bit_flip_tf(tf.zeros_like(discriminator_fake[0]))    #fake=0
    loss_true = tf.reduce_mean(- label_true * tf.math.log(discriminator_true[0] + 2e-7) - 
                               (1 - label_true) *tf.math.log(1 - discriminator_true[0] + 2e-7))
    loss_fake = tf.reduce_mean(- label_fake * tf.math.log(discriminator_fake[0] + 2e-7) - 
                               (1 - label_fake) *tf.math.log(1 - discriminator_fake[0] + 2e-7))
    loss_true_fake = (loss_true + loss_fake)/2.


    #aux loss
    loss_aux_true = tf.reduce_mean(tf.math.abs((energy_batch - discriminator_true[1])/(energy_batch + 2e-7))) *100
    loss_aux_fake = tf.reduce_mean(tf.math.abs((gen_aux - discriminator_fake[1])/(gen_aux + 2e-7))) *100
    loss_aux = (loss_aux_true + loss_aux_fake)/2.
    
    #ecal loss
    loss_ecal_true = tf.reduce_mean(tf.math.abs((ecal_batch - discriminator_true[2])/(ecal_batch + 2e-7))) *100
    loss_ecal_fake = tf.reduce_mean(tf.math.abs((gen_ecal - discriminator_fake[2])/(gen_ecal + 2e-7))) *100
    loss_ecal = (loss_ecal_true + loss_ecal_fake)/2.

    #total loss
    weight_true_fake = 6.0
    weight_aux = 0.2
    weight_ecal = 0.1
    total_loss = weight_true_fake * loss_true_fake + weight_aux * loss_aux + weight_ecal * loss_ecal
    return total_loss, loss_true_fake, loss_aux, loss_ecal


def gen_loss(g_weights, d_weights, batch_size, latent_size=200):
    noise, gen_aux, generator_input, gen_ecal = func_for_gen_tf(nb_test=batch_size, latent_size=latent_size) #len has to be batch size, because of bit_flip used below
    generated_images = generator(generator_input, g_weights)
    discriminator_fake = discriminator(generated_images, d_weights)

    #true/fake
    label_fake = bit_flip_tf(tf.ones_like(discriminator_fake[0]))   #ones = true
    loss_fake = tf.reduce_mean(- label_fake * tf.math.log(discriminator_fake[0] + 2e-7) - 
                               (1 - label_fake) * tf.math.log(1 - discriminator_fake[0] + 2e-7))
    loss_true_fake = loss_fake
    
    #aux
    loss_aux_fake = tf.reduce_mean(tf.math.abs((gen_aux - discriminator_fake[1])/(gen_aux + 2e-7))) *100
    loss_aux = loss_aux_fake
    
    #ecal
    loss_ecal_fake = tf.reduce_mean(tf.math.abs((gen_ecal - discriminator_fake[2])/(gen_ecal + 2e-7))) *100
    loss_ecal = loss_ecal_fake
    
    #total loss
    weight_true_fake = 6.0
    weight_aux = 0.2
    weight_ecal = 0.1
    total_loss = weight_true_fake * loss_true_fake + weight_aux * loss_aux + weight_ecal * loss_ecal
    return total_loss, loss_true_fake, loss_aux, loss_ecal












