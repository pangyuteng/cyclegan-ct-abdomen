# source:
# https://github.com/keras-team/keras-io/blob/master/examples/generative/cyclegan.py 
#
import os
import random
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

def seed_everything(seed=4269):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
seed_everything()

'''
#from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
CUDA_VISIBLE_DEVICES=3 python cyclegan_resnet.py
'''
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os

from ganutils import get_resnet_generator, get_discriminator
from tbutils import ImageSummaryCallback, MetricSummaryCallback

class CycleGAN():
    def __init__(self):
        
        # logging
        self.log_dir = './log'
        self.image_summary_callback = ImageSummaryCallback(self.log_dir)
        self.metric_summary_callback = MetricSummaryCallback(self.log_dir)
        
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'c4kc-kits'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols),augment=True)


        # Calculate output shape of D (PatchGAN)
        self.var = 3
        patch = int(self.img_rows / 2**self.var)
        self.disc_patch = (patch, patch, 1)

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

        #optimizer = Adam(0.0002, 0.5)
        optimizer = Adam(0.0001, 0.4)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator(name="d_A")
        self.d_B = self.build_discriminator(name="d_B")
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator(name="g_AB")
        self.g_BA = self.build_generator(name="g_BA")

        if True:
            print("self.g_AB")
            self.g_AB.summary()
            print("self.d_A")
            self.d_A.summary()
        
        w_list = ['saved_model/dA.h5','saved_model/dB.h5','saved_model/AB.h5','saved_model/BA.h5']
        if all([os.path.exists(x) for x in w_list]):
            print('found weights, loading them...')
            self.d_A.load_weights("saved_model/dA.h5")
            self.d_B.load_weights("saved_model/dB.h5")
            self.g_AB.load_weights("saved_model/AB.h5")
            self.g_BA.load_weights("saved_model/BA.h5")

        # Input images from both domains
        img_A = keras.layers.Input(shape=self.img_shape)
        img_B = keras.Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)

    def build_generator(self,name):
        kwargs = dict(
            filters=64,
            num_downsampling_blocks=self.var,
            num_residual_blocks=12,
            num_upsample_blocks=self.var,            
            input_img_size=self.img_shape
        )
        return get_resnet_generator(name=name,**kwargs)

    def build_discriminator(self,name):
        kwargs = dict(
            filters=64,
            num_downsampling=3,
            input_img_size=self.img_shape,
        )
        return get_discriminator(name=name,**kwargs)

    def train(self, epochs, batch_size=1, sample_interval=50):
        
        start_time = datetime.datetime.now()
        '''
        if np.random.random() > 0.5:
            batch_size = 1
        else:
            batch_size = 5
        '''
        
        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                print(imgs_A.shape,imgs_B.shape)
                print('!!!!!!!!!!!!!!!!!!!11')
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                        [valid, valid,
                                                        imgs_A, imgs_B,
                                                        imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))

                # logging
                mydict = dict(
                    d_loss = d_loss[0],
                    g_loss = g_loss[0],
                    adv = np.mean(g_loss[1:3]),
                    recon = np.mean(g_loss[3:5]),
                    id = np.mean(g_loss[5:6]),
                )
                self.metric_summary_callback.on_epoch_end(epoch,mydict=mydict)

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
                    os.makedirs("saved_model",exist_ok=True)
                    self.g_AB.save_weights("saved_model/AB.h5")
                    self.g_BA.save_weights("saved_model/BA.h5")
                    self.d_A.save_weights("saved_model/dA.h5")
                    self.d_B.save_weights("saved_model/dB.h5")
          
    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 3

        imgs_A = self.data_loader.load_data(domain="noncontrast", batch_size=1)
        imgs_B = self.data_loader.load_data(domain="arterial", batch_size=1)

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()
        self.image_summary_callback.on_epoch_end(epoch,mydict={"img":gen_imgs*255})


if __name__ == '__main__':
    gan = CycleGAN()    
    gan.train(epochs=200, batch_size=1, sample_interval=200)
