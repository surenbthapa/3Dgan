#my own file, the functions are from: https://github.com/svalleco/3Dgan/blob/Energy-gan/keras/analysis/utils/GANutils.py

import os
import h5py
import numpy as np
import math
import time
import glob
import numpy.core.umath_tests as umath
try:
    import cPickle as pickle
except ImportError:
    import pickle
import tensorflow as tf


import numpy.core.umath_tests as umath

def get_moments(sumsx, sumsy, sumsz, totalE, m, x=51, y=51, z=25):
    old_err_state = np.seterr(divide='raise')
    ignored_states = np.seterr(**old_err_state)
    totalE = np.squeeze(totalE)
    index = sumsx.shape[0]
    momentX = np.zeros((index, m))
    momentY = np.zeros((index, m))
    momentZ = np.zeros((index, m))
    ECAL_midX = np.zeros(index)
    ECAL_midY = np.zeros(index)
    ECAL_midZ = np.zeros(index)
    for i in range(m):
      relativeIndices = np.tile(np.arange(x), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midX).transpose(), i+1)
      ECAL_momentX = np.divide(umath.inner1d(sumsx, moments) ,totalE)
      if i==0: ECAL_midX = ECAL_momentX.transpose()
      momentX[:,i] = ECAL_momentX
    for i in range(m):
      relativeIndices = np.tile(np.arange(y), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midY).transpose(), i+1)
      ECAL_momentY = np.divide(umath.inner1d(sumsy, moments), totalE)
      if i==0: ECAL_midY = ECAL_momentY.transpose()
      momentY[:,i]= ECAL_momentY
    for i in range(m):
      relativeIndices = np.tile(np.arange(z), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midZ).transpose(), i+1)
      ECAL_momentZ = np.divide(umath.inner1d(sumsz, moments), totalE)
      if i==0: ECAL_midZ = ECAL_momentZ.transpose()
      momentZ[:,i]= ECAL_momentZ
    return momentX, momentY, momentZ


# find location of maximum depositions
def get_max(images):
    index = images.shape[0]
    x=images.shape[1]
    y=images.shape[2]
    z=images.shape[3]
    max_pos = np.zeros((index, 3))
    for i in range(index):
       max_p = images[i].argmax()
       max_loc = np.unravel_index(max_p, (x, y, z))
       max_pos[i] = max_loc
    return max_pos

def GetData(datafile, thresh=0):
   #get data for training
    print( 'Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    y=f.get('target')
    x=np.array(f.get('ECAL'))
    y=(np.array(y[:,1]))
    if thresh>0:
       x[x < thresh] = 0
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return x, y

# get sums along different axis
def get_sums(images):
    sumsx = np.squeeze(np.sum(images, axis=(2,3)))
    sumsy = np.squeeze(np.sum(images, axis=(1,3)))
    sumsz = np.squeeze(np.sum(images, axis=(1,2)))
    return sumsx, sumsy, sumsz

def get_sorted(datafiles, energies, flag=False, num_events1=10000, num_events2=2000, tolerance=5, thresh=1e-6):
    srt = {}
    for index, datafile in enumerate(datafiles):
        data = GetData(datafile, thresh)
        X = data[0]
        sumx = np.sum(np.squeeze(X), axis=(1, 2, 3))
        indexes= np.where(sumx>0)
        X=X[indexes]
        Y = data[1]
        Y=Y[indexes]
        for energy in energies:
            if index== 0:
                if energy == 0:
                    srt["events_act" + str(energy)] = X # More events in random bin
                    srt["energy" + str(energy)] = Y
                    if srt["events_act" + str(energy)].shape[0] > num_events1:
                        srt["events_act" + str(energy)] = srt["events_act" + str(energy)][:num_events1]
                        srt["energy" + str(energy)] = srt["energy" + str(energy)][:num_events1]
                        flag=False
                else:
                    indexes = np.where((Y > energy - tolerance ) & ( Y < energy + tolerance))
                    srt["events_act" + str(energy)] = X[indexes]
                    srt["energy" + str(energy)] = Y[indexes]
            else:
                if energy == 0:
                   if flag:
                    srt["events_act" + str(energy)] = np.append(srt["events_act" + str(energy)], X, axis=0)
                    srt["energy" + str(energy)] = np.append(srt["energy" + str(energy)], Y, axis=0)
                    if srt["events_act" + str(energy)].shape[0] > num_events1:
                        srt["events_act" + str(energy)] = srt["events_act" + str(energy)][:num_events1]
                        srt["energy" + str(energy)] = srt["energy" + str(energy)][:num_events1]
                        flag=False
                else:
                    if srt["events_act" + str(energy)].shape[0] < num_events2:
                        indexes = np.where((Y > energy - tolerance ) & ( Y < energy + tolerance))
                        srt["events_act" + str(energy)] = np.append(srt["events_act" + str(energy)], X[indexes], axis=0)
                        srt["energy" + str(energy)] = np.append(srt["energy" + str(energy)], Y[indexes], axis=0)
                    srt["events_act" + str(energy)] = srt["events_act" + str(energy)][:num_events2]
                    srt["energy" + str(energy)] = srt["energy" + str(energy)][:num_events2]
    return srt


def safe_mkdir(path):
   #Safe mkdir (i.e., don't create if already exists,and no violation of race conditions)
    from os import makedirs
    from errno import EEXIST
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != EEXIST:
            raise exception

def DivideFiles(FileSearch="/data/LCD/*/*.h5",
                Fractions=[.9,.1],datasetnames=["ECAL","HCAL"],Particles=[],MaxFiles=-1):
    print ("Searching in :",FileSearch)
    Files =sorted( glob.glob(FileSearch))
    print ("Found {} files. ".format(len(Files)))
    FileCount=0
    Samples={}
    for F in Files:
        FileCount+=1
        basename=os.path.basename(F)
        ParticleName=basename.split("_")[0].replace("Escan","")
        if ParticleName in Particles:
            try:
                Samples[ParticleName].append(F)
            except:
                Samples[ParticleName]=[(F)]
        if MaxFiles>0:
            if FileCount>MaxFiles:
                break
    out=[]
    for j in range(len(Fractions)):
        out.append([])
    SampleI=len(Samples.keys())*[int(0)]
    for i,SampleName in enumerate(Samples):
        Sample=Samples[SampleName]
        NFiles=len(Sample)
        for j,Frac in enumerate(Fractions):
            EndI=int(SampleI[i]+ round(NFiles*Frac))
            out[j]+=Sample[SampleI[i]:EndI]
            SampleI[i]=EndI
    return out

# generate images
def generate(g, gen_weights, index, cond, latent=256, concat=1):
    energy_labels=np.expand_dims(cond[0], axis=1)
    if len(cond)> 1: # that means we also have angle
      angle_labels = cond[1]
      if concat==1:
        noise = np.random.normal(0, 1, (index, latent-1))  
        noise = energy_labels * noise
        gen_in = np.concatenate((angle_labels.reshape(-1, 1), noise), axis=1)
      elif concat==2:
        noise = np.random.normal(0, 1, (index, latent-2))
        gen_in = np.concatenate((energy_labels, angle_labels.reshape(-1, 1), noise), axis=1)
      else:  
        noise = np.random.normal(0, 1, (index, 2, latent))
        angle_labels=np.expand_dims(angle_labels, axis=1)
        gen_in = np.concatenate((energy_labels, angle_labels), axis=1)
        gen_in = np.expand_dims(gen_in, axis=2)
        gen_in = gen_in * noise
    else:
      print("else")
      noise = np.random.normal(0, 1, (index, latent))
      #energy_labels=np.expand_dims(energy_labels, axis=1)
      gen_in = energy_labels * noise
    generated_images = g.predict(gen_in, gen_weights)
    return generated_images

def tf_generate(generator, gen_weights, index, cond, latent=200, concat=1):
    energy_labels =     np.expand_dims(cond[0], axis=1)
    print("len_labels: ", len(energy_labels))
    batch_size = 50
    iterations = int(len(energy_labels) / batch_size)
    energy_labels =    tf.constant(energy_labels)
    noise =            tf.Variable(tf.random.normal([index,latent], mean=0.0, stddev=1.0)) 
    gen_in =           energy_labels * noise
    #generated_images = [0]
    generator_in = gen_in[0*batch_size : (1)*batch_size]
    generated_images = generator(generator_in, gen_weights)
    for iteration in range(1,iterations+1):
        generator_in = gen_in[iteration*batch_size : (iteration+1)*batch_size]
        gen_out = generator(generator_in, gen_weights)
        generated_images = tf.concat([generated_images, gen_out], axis = 0)
    print("gen_images: ", generated_images.shape)
    return generated_images


def discriminate(d, images):
    disc_out = d.predict(images, verbose=False, batch_size=50)
    return disc_out


def tf_discriminate(d, images, g_weights, d_weights):
    print("len images: ", len(images))
    batch_size = 50
    iterations = int(len(images) / batch_size)
    discriminator_out = d(images[0:batch_size], g_weights, d_weights)
    print("disc_out: ", discriminator_out)
    for iteration in range(1,iterations+1):
        disc_in = images[iteration*batch_size : (iteration+1)*batch_size]
        disc_out = d(disc_in, g_weights, d_weights)
        discriminator_out = tf.concat([discriminator_out, disc_out], axis = 1)
    print("disc_out shape: ", np.array(discriminator_out).shape)
    return discriminator_out



def perform_calculations_multi(g, d, gweights, dweights, energies, datapath, sortdir, gendirs, discdirs, num_data, 
                               num_events, m, scales, thresh, latent, events_per_file=10000, particle='Ele', dformat='channels_last'):
    sortedpath = os.path.join(sortdir, 'events_*.h5')   #dont know what this is
    var= {}
    num_events1= 10000
    num_events2 = num_events
    read_data = False    #i created this
    Test = False    #i created this
    save_data = False
    read_gen = False
    read_disc = False
    save_gen = False
    save_disc = False
    if read_data: # Read from sorted dir                                                                                                                                                                           
       start = time.time()
       var = load_sorted(sortedpath, energies)
       sort_time = time.time()- start
       print ("Events were loaded in {} seconds".format(sort_time))
    else:
       # Getting Data                                                                                                                                                                                              
       events_per_file = 10000
       Filesused = int(math.ceil(num_data/events_per_file))
       print(Filesused)
       Trainfiles, Testfiles = DivideFiles(datapath, datasetnames=["ECAL"], Particles =[particle])
       Trainfiles = Trainfiles[: Filesused]
       Testfiles = Testfiles[: Filesused]
       print(len(Trainfiles))
       print(len(Testfiles))
       
       if Test:
          data_files = Testfiles
       else:
          data_files = Trainfiles
       start = time.time()
       print(data_files)
       var = get_sorted(data_files, energies, True, num_events1, num_events2)
       data_time = time.time() - start
       print ("{} events were loaded in {} seconds".format(num_data, data_time))
       if save_data:
          save_sorted(var, energies, sortdir)
    total = 0
    for energy in energies:
 #       print("checkpoint1 :", energy)
        #calculations for data events
        var["events_act"+ str(energy)]= np.squeeze(var["events_act"+ str(energy)])
        # Getting dimensions of ecal images
        x = var["events_act"+ str(energy)].shape[1]
        y =var["events_act"+ str(energy)].shape[2]
        z =var["events_act"+ str(energy)].shape[3]

        var["index" + str(energy)]= var["energy" + str(energy)].shape[0]
        total += var["index" + str(energy)]
        var["ecal_act"+ str(energy)]=np.sum(var["events_act"+ str(energy)], axis=(1, 2, 3))
        var["max_pos_act" + str(energy)] = get_max(var["events_act" + str(energy)])
        var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)] = get_sums(var["events_act" + str(energy)])
        var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)]= get_moments(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], var["ecal_act"+ str(energy)], m, x=x, y=y, z=z)
#    print("checkpoint2")
    data_time = time.time() - start
    print ("{} events were put in {} bins".format(total, len(energies)))
    #### Generate Data table to screen                                                                                                                                                                             
    print ("Actual Data")
    print ("Energy\tEvents\tMaximum Value\t\t\tMaximum loc\tMean\t\tMomentx2\tMomenty2\tMomentz2")
    for energy in energies:
#        print("checkpoint3 :", energy)
        print ("{}\t{}\t{:.4f}\t\t{}\t{:.2f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}" .format(energy, var["index" +str(energy)], np.amax(var["events_act" + str(energy)]), np.mean(var["max_pos_act" + str(energy)], axis=0), np.mean(var["events_act" + str(energy)]), np.mean(var["momentX_act"+ str(energy)][:, 1]), np.mean(var["momentY_act"+ str(energy)][:, 1]), np.mean(var["momentZ_act"+ str(energy)][:, 1])))

  #  for gen_weights, disc_weights, scale, i in zip(gweights, dweights, scales, np.arange(len(gweights))):
    for i in np.arange(len(gweights)):
       #gen_weights = gweights[0]
       #disc_weights = dweights[0]
       disc_weights = pickle.load( open(dweights[0], "rb") )
       gen_weights = pickle.load( open(gweights[0], "rb") )
       scale = scales[0]
#       print(gen_weights, disc_weights)                             #
       gendir = gendirs + '/n_' + str(i)
       discdir = discdirs + '/n_' + str(i)
       for energy in energies:
                               
          var["events_gan" + str(energy)]={}
          var["isreal_act" + str(energy)]={}
          var["isreal_gan" + str(energy)]={}
          var["aux_act" + str(energy)]={}
          var["aux_gan" + str(energy)]={}
          var["ecal_act" + str(energy)]={}
          var["ecal_gan" + str(energy)]={}
          var["max_pos_gan" + str(energy)]={}
          var["sumsx_gan"+ str(energy)]={}
          var["sumsy_gan"+ str(energy)]={}
          var["sumsz_gan"+ str(energy)]={}
          var["momentX_gan" + str(energy)]={}
          var["momentY_gan" + str(energy)]={}
          var["momentZ_gan" + str(energy)]={}
          if read_gen:
              var["events_gan" + str(energy)]['n_'+ str(i)]= get_gen(energy, gendir)
          else:
              start = time.time()
              #var["events_gan" + str(energy)]['n_'+ str(i)] = generate(g, gen_weights, var["index" + str(energy)], [var["energy" + str(energy)]/100], latent)
              var["events_gan" + str(energy)]['n_'+ str(i)] = tf_generate(g, gen_weights, var["index" + str(energy)], [var["energy" + str(energy)]/100], latent)
              var["events_gan" + str(energy)]['n_'+ str(i)] = np.squeeze(var["events_gan" + str(energy)]['n_'+ str(i)])
              if save_gen:
                  save_generated(var["events_gan" + str(energy)]['n_'+ str(i)], var["energy" + str(energy)], energy, gendir)
              gen_time = time.time() - start
              #print("checkpoint 4")
              print ("Generator took {} seconds to generate {} events".format(gen_time, var["index" +str(energy)]))
          
        
        
        
        #discriminate images
          if read_disc:
             var["isreal_act" + str(energy)]['n_'+ str(i)], var["aux_act" + str(energy)]['n_'+ str(i)], var["ecal_act"+ str(energy)]['n_'+ str(i)], var["isreal_gan" + str(energy)]['n_'+ str(i)], var["aux_gan" + str(energy)]['n_'+ str(i)], var["ecal_gan"+ str(energy)]['n_'+ str(i)]= get_disc(energy, discdir)
 
          else:
             start = time.time()
             if dformat=='channels_last':
                 var["events_act" + str(energy)] = np.expand_dims(var["events_act" + str(energy)], axis=-1)
                 var["events_gan" + str(energy)]['n_'+ str(i)] = np.expand_dims(var["events_gan" + str(energy)]['n_'+ str(i)], axis=-1)
             else:
                var["events_act" + str(energy)] = np.expand_dims(var["events_act" + str(energy)], axis=1)
                var["events_gan" + str(energy)]['n_'+ str(i)] = np.expand_dims(var["events_gan" + str(energy)]['n_'+ str(i)], axis=1)
             discout= tf_discriminate(d, var["events_act" + str(energy)] * scale, gen_weights, disc_weights)
             
             var["isreal_act" + str(energy)]['n_'+ str(i)], var["aux_act" + str(energy)]['n_'+ str(i)], var["ecal_act"+ str(energy)]['n_'+ str(i)]= tf_discriminate(d, var["events_act" + str(energy)] * scale, gen_weights, disc_weights)
             var["isreal_gan" + str(energy)]['n_'+ str(i)], var["aux_gan" + str(energy)]['n_'+ str(i)], var["ecal_gan"+ str(energy)]['n_'+ str(i)]= tf_discriminate(d, var["events_gan" + str(energy)]['n_'+ str(i)] , gen_weights, disc_weights)
             disc_time = time.time() - start
             print ("Discriminator took {} seconds for {} data and generated events".format(disc_time, var["index" +str(energy)]))
             var["events_act" + str(energy)]= np.squeeze(var["events_act" + str(energy)])
             var["events_gan" + str(energy)]['n_'+ str(i)]= np.squeeze(var["events_gan" + str(energy)]['n_'+ str(i)])
             if save_disc:
               discout = {}
               for key in var:
                  if key in ["isreal_act" + str(energy), "aux_act" + str(energy), "isreal_gan" + str(energy), "aux_gan" + str(energy), "ecal_act"+ str(energy), "ecal_gan"+ str(energy)]:
                     discout[key]=var[key]['n_'+ str(i)]
               for key in discout:
                   print (key)
               save_discriminated(discout, energy, discdir)
          print ('Calculations for ....', energy)
          var["events_gan" + str(energy)]['n_'+ str(i)] = var["events_gan" + str(energy)]['n_'+ str(i)]/scale
          var["isreal_act" + str(energy)]['n_'+ str(i)] = np.squeeze(var["isreal_act" + str(energy)]['n_'+ str(i)])
          var["isreal_act" + str(energy)]['n_'+ str(i)], var["aux_act" + str(energy)]['n_'+ str(i)], var["ecal_act"+ str(energy)]['n_'+ str(i)]= np.squeeze(var["isreal_act" + str(energy)]['n_'+ str(i)]), np.squeeze(var["aux_act" + str(energy)]['n_'+ str(i)]), np.squeeze(var["ecal_act"+ str(energy)]['n_'+ str(i)]/scale)

          var["isreal_gan" + str(energy)]['n_'+ str(i)], var["aux_gan" + str(energy)]['n_'+ str(i)], var["ecal_gan"+ str(energy)]['n_'+ str(i)]= np.squeeze(var["isreal_gan" + str(energy)]['n_'+ str(i)]), np.squeeze(var["aux_gan" + str(energy)]['n_'+ str(i)]), np.squeeze(var["ecal_gan"+ str(energy)]['n_'+ str(i)]/scale)
          var["max_pos_gan" + str(energy)]['n_'+ str(i)] = get_max(var["events_gan" + str(energy)]['n_'+ str(i)])
          var["sumsx_gan"+ str(energy)]['n_'+ str(i)], var["sumsy_gan"+ str(energy)]['n_'+ str(i)], var["sumsz_gan"+ str(energy)]['n_'+ str(i)] = get_sums(var["events_gan" + str(energy)]['n_'+ str(i)])
          var["momentX_gan" + str(energy)]['n_'+ str(i)], var["momentY_gan" + str(energy)]['n_'+ str(i)], var["momentZ_gan" + str(energy)]['n_'+ str(i)] = get_moments(var["sumsx_gan"+ str(energy)]['n_'+ str(i)], var["sumsy_gan"+ str(energy)]['n_'+ str(i)], var["sumsz_gan"+ str(energy)]['n_'+ str(i)], var["ecal_gan"+ str(energy)]['n_'+ str(i)], m, x=x, y=y, z=z)

#       print('For {} iteration:\nWith Generator weights.....{}\nWith Discriminator weights.....{}'.format(i, gen_weights, disc_weights))

       #### Generate GAN table to screen                                                                                                                                                                          
 
       print ("Generated Data")
       print ("Energy\tEvents\tMaximum Value\t\t\tMaximum loc\t\t\tMean\t\tMomentx2\tMomenty2\tMomentz2")

       for energy in energies:
          print ("{}\t{}\t{:.4f}\t\t{}\t\t\t{:.2f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}".format(energy, var["index" +str(energy)], np.amax(var["events_gan" + str(energy)]['n_'+ str(i)]), np.mean(var["max_pos_gan" + str(energy)]['n_'+ str(i)], axis=0), np.mean(var["events_gan" + str(energy)]['n_'+ str(i)]), np.mean(var["momentX_gan"+ str(energy)]['n_'+ str(i)][:, 1]), np.mean(var["momentY_gan"+ str(energy)]['n_'+ str(i)][:, 1]), np.mean(var["momentZ_gan"+ str(energy)]['n_'+ str(i)][:, 1])))

    return var


