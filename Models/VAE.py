import numpy as np
import pandas as pd
import os.path
import pickle

from Models.RBM_TF_2 import RBM, RBM_Linear

from keras.layers import Input, Dense, Flatten, Lambda, Reshape
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint 

class VariationalAutoencoder:

    kind = 'VariationalAutoencoder'

    def __init__(self,layer_dims):
        '''
            Inputs:

            - layer_dims = A list of the layer sizes, visible first, latent last

            Note that the number of hidden layers in the unrolled 
            variational autoencoder will be twice the length of layer_dims. 
        '''

        self.latent_dim = layer_dims[-1]
        self.v_dim      = layer_dims[0]
        self.num_hidden_layers = len(layer_dims)-2
        self.layer_dims = layer_dims[0:-1]          

        self.W = []
        self.b = []
        self.a = []
        self.pretrained = False
        return

    @classmethod
    def pretrained_from_file(cls,filename):
        '''
            Initialize with pretrained weights from a file.

            Still needs to be unrolled.
        '''
        i = 0
        weights = []
        layer_dims = []

        while os.path.isfile(filename+"_"+str(i)+"_a.csv"): # load the next layer's weights
            weights.append(RBM.load_weights(filename+"_"+str(i))) # load the next dict of weights
            layer_dims.append(np.shape(weights[i]['W'])[0])
            i = i+1
        layer_dims.append(np.shape(weights[i-1]['W'])[1])

        rbm = cls(layer_dims)

        for i in range(rbm.num_hidden_layers): 
            rbm.W.append(weights[i]['W'])
            rbm.a.append(weights[i]['a'])
            rbm.b.append(weights[i]['b'])
        
        rbm.pretrained = True

        return rbm

    def pretrain(self,x,epochs,num_samples = 50000):
        '''
            Greedy layer-wise training
            
            The last layer is a RBM with linear hidden units

            shape(x) = (v_dim, number_of_examples)
        '''
        print('Pretraining \n')

        RBM_layers = []

        for i in range(self.num_hidden_layers): # initialize RBM's
            if (i < self.num_hidden_layers - 1):
                RBM_layers.append(RBM(self.layer_dims[i],self.layer_dims[i+1]))
            
            else:
                RBM_layers.append(RBM_Linear(self.layer_dims[i],self.layer_dims[i+1]))
        
        for i in range(self.num_hidden_layers):  # train RBM's 
            print(f"Training RBM layer {i+1}, size: {self.layer_dims[i]}")

            RBM_layers[i].train(x,epochs) # train the ith RBM
            
            if not(i == self.num_hidden_layers - 1): # generate samples to train next layer
                
                _ , x = RBM_layers[i].gibbs_sampling(2,num_samples) 

            _W, _a, _b = RBM_layers[i].get_weights()

            self.W.append(_W) # save trained weights
            self.b.append(_b)
            self.a.append(_a)

        self.pretrained = True

        return

    
    def unroll(self):
        '''
            Unrolls the pretrained RBM network into a DFF keras model 
            and sets hidden layer parameters to pretrained values.

            Returns the keras model
        '''
        if self.pretrained == False:
            
            print("Model not pretrained.")
            return

        # define keras model structure
        encoder_input = Input(shape=(self.v_dim,), name='encoder_input')
        
        x = encoder_input
        
        # build encoder 
        for i in range(self.num_hidden_layers):            
            weights = [self.W[i],self.b[i].flatten()]
            
            if (i == self.num_hidden_layers - 1):
                dense_layer = Dense(self.layer_dims[i+1],  
                                    weights = weights, 
                                    name = 'encoder_dense_' + str(i))
                x = dense_layer(x)
            
            else:
                dense_layer = Dense(self.layer_dims[i+1], 
                                    activation='sigmoid',
                                    weights = weights,
                                    name = 'encoder_dense_' + str(i))
                x = dense_layer(x)

        #Build the Latent Sampling
        shape_before_flattening = K.int_shape(x)[1:]
        
        self.mu = Dense(self.latent_dim, name='mu')(x)
        self.log_var = Dense(self.latent_dim, name='log_var')(x)

        self.encoder_mu_log_var = Model(encoder_input, (self.mu, self.log_var))

        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon

        encoder_output = Lambda(sampling, name='encoder_output')([self.mu, self.log_var])

        self.encoder = Model(encoder_input, encoder_output)

        
         ### THE DECODER

        decoder_input = Input(shape=(self.latent_dim,), name='decoder_input')

        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        #x = Reshape(shape_before_flattening)(x)        

        # build decoder
      
        for i in range(self.num_hidden_layers):
            weights = [self.W[self.num_hidden_layers-i-1].T,self.a[self.num_hidden_layers-i-1].flatten()]

            dense_layer = Dense(self.layer_dims[self.num_hidden_layers-i-1],
                      activation='sigmoid', 
                      weights = weights,
                      name = 'decoder_dense_' + str(i))                        
            
            x = dense_layer(x)
        
        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output)

        ### THE FULL VAE
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)

    def compile(self, learning_rate, r_loss_factor, Beta):
        """
        Compiling the network. Need to choose the learning rate, r_loss_factor
        and Beta for the Beta-VAE, if Beta = 1 then it is a VAE.
        
        Parameters
        ----------------------------------------------------------------------
        learning_rate: Learning Rate for gradient descent.
        r_loss_factor: Factor that multiplies the loss factor of the
                       reconstruction loss.
        Beta: Beta-VAE parameter that multiplies the KL-Divergence in order to
              disentangle the latent space of the model.
              
        """
        
        self.learning_rate = learning_rate

        ### COMPILATION
        def vae_r_loss(y_true, y_pred):
            r_loss = K.mean(K.square(y_true - y_pred), axis = -1)
            return r_loss_factor * r_loss

        def vae_kl_loss(y_true, y_pred):
            kl_loss =  -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis = 1)
            return kl_loss

        def vae_loss(y_true, y_pred):
            r_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(y_true, y_pred)
            return  r_loss + Beta*kl_loss

        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss = vae_loss,  metrics = [vae_r_loss, vae_kl_loss])

    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches = 100, initial_epoch = 0, lr_decay = 1):
        
        # custom_callback = CustomCallback(run_folder, print_every_n_batches, initial_epoch, self)
        # lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint_filepath=os.path.join(run_folder, "weights/weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = ModelCheckpoint(checkpoint_filepath, save_weights_only = True, verbose=0)
        checkpoint2 = ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'), save_weights_only = True, verbose=0)

        callbacks_list = [checkpoint1, checkpoint2]

        self.model.fit(     
            x_train
            , x_train
            , batch_size = batch_size
            , shuffle = True
            , epochs = epochs
            , initial_epoch = initial_epoch
            , callbacks = callbacks_list
        )


    def save(self,filename):
        '''
            saves the pretrained weights. Saving and loading a keras model 
            after pretraining is better done directly to the self.autoencoder
            object using the keras fnctions save() and load_model()
        '''

        if self.pretrained == True:
            for i in range(self.num_hidden_layers):
                weights = {"W":self.W[i],'a':self.a[i],'b':self.b[i]}
                RBM.save_weights(weights,filename+"_"+str(i))
        else: 
            print("No pretrained weights to save.")

        return    

class VariationalAutoencoderEntanglement:

    kind = 'VariationalAutoencoder'

    def __init__(self,layer_dims):
        '''
            Inputs:

            - layer_dims = A list of the layer sizes, visible first, latent last

            Note that the number of hidden layers in the unrolled 
            variational autoencoder will be twice the length of layer_dims. 
        '''

        self.latent_dim = layer_dims[-1]
        self.v_dim      = layer_dims[0]
        self.num_hidden_layers = len(layer_dims)-2
        self.layer_dims = layer_dims[0:-1]          

        self.W = []
        self.b = []
        self.a = []
        self.pretrained = False
        return

    @classmethod
    def pretrained_from_file(cls,filename):
        '''
            Initialize with pretrained weights from a file.

            Still needs to be unrolled.
        '''
        i = 0
        weights = []
        layer_dims = []

        while os.path.isfile(filename+"_"+str(i)+"_a.csv"): # load the next layer's weights
            weights.append(RBM.load_weights(filename+"_"+str(i))) # load the next dict of weights
            layer_dims.append(np.shape(weights[i]['W'])[0])
            i = i+1
        layer_dims.append(np.shape(weights[i-1]['W'])[1])

        rbm = cls(layer_dims)

        for i in range(rbm.num_hidden_layers): 
            rbm.W.append(weights[i]['W'])
            rbm.a.append(weights[i]['a'])
            rbm.b.append(weights[i]['b'])
        
        rbm.pretrained = True

        return rbm

    def pretrain(self,x,epochs,num_samples = 50000):
        '''
            Greedy layer-wise training
            
            The last layer is a RBM with linear hidden units

            shape(x) = (v_dim, number_of_examples)
        '''
        print('Pretraining \n')

        RBM_layers = []

        for i in range(self.num_hidden_layers): # initialize RBM's
            if (i < self.num_hidden_layers - 1):
                RBM_layers.append(RBM(self.layer_dims[i],self.layer_dims[i+1]))
            
            else:
                RBM_layers.append(RBM_Linear(self.layer_dims[i],self.layer_dims[i+1]))
        
        for i in range(self.num_hidden_layers):  # train RBM's 
            print(f"Training RBM layer {i+1}, size: {self.layer_dims[i]}")

            RBM_layers[i].train(x,epochs) # train the ith RBM
            
            if not(i == self.num_hidden_layers - 1): # generate samples to train next layer
                
                _ , x = RBM_layers[i].gibbs_sampling(2,num_samples) 

            _W, _a, _b = RBM_layers[i].get_weights()

            self.W.append(_W) # save trained weights
            self.b.append(_b)
            self.a.append(_a)

        self.pretrained = True

        return

    
    def unroll(self):
        '''
            Unrolls the pretrained RBM network into a DFF keras model 
            and sets hidden layer parameters to pretrained values.

            Returns the keras model
        '''
        if self.pretrained == False:
            
            print("Model not pretrained.")
            return

        # define keras model structure
        encoder_input = Input(shape=(self.v_dim,), name='encoder_input')
        
        x = encoder_input
        
        # build encoder 
        for i in range(self.num_hidden_layers):            
            weights = [self.W[i],self.b[i].flatten()]
            
            if (i == self.num_hidden_layers - 1):
                dense_layer = Dense(self.layer_dims[i+1],  
                                    weights = weights, 
                                    name = 'encoder_dense_' + str(i))
                x = dense_layer(x)
            
            else:
                dense_layer = Dense(self.layer_dims[i+1], 
                                    activation='sigmoid',
                                    weights = weights,
                                    name = 'encoder_dense_' + str(i))
                x = dense_layer(x)

        #Build the Latent Sampling
        shape_before_flattening = K.int_shape(x)[1:]
        
        self.mu = Dense(self.latent_dim, name='mu')(x)
        self.log_var = Dense(self.latent_dim, name='log_var')(x)

        self.encoder_mu_log_var = Model(encoder_input, (self.mu, self.log_var))

        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon

        encoder_output = Lambda(sampling, name='encoder_output')([self.mu, self.log_var])

        self.encoder = Model(encoder_input, encoder_output)

        
         ### THE DECODER

        decoder_input = Input(shape=(self.latent_dim,), name='decoder_input')

        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        

        # build decoder
      
        for i in range(self.num_hidden_layers):
            weights = [self.W[self.num_hidden_layers-i-1].T,self.a[self.num_hidden_layers-i-1].flatten()]

            #Adding a linear output unit
            if (i == self.num_hidden_layers - 2):
                dense_layer = Dense(self.layer_dims[self.num_hidden_layers-i-1],  
                                    weights = weights, 
                                    name = 'dencoder_dense_' + str(i))
                x = dense_layer(x)

            else:
              dense_layer = Dense(self.layer_dims[self.num_hidden_layers-i-1],
                        activation='sigmoid', 
                        weights = weights,
                        name = 'dcoder_dense_' + str(i))                        
            
            x = dense_layer(x)

        #Adding a softmax output
        #x = Dense(2, activation='softmax')(x)

        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output)

        ### THE FULL VAE
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)

    def compile(self, learning_rate, r_loss_factor, Beta):
        """
        Compiling the network. Need to choose the learning rate, r_loss_factor
        and Beta for the Beta-VAE, if Beta = 1 then it is a VAE.
        
        Parameters
        ----------------------------------------------------------------------
        learning_rate: Learning Rate for gradient descent.
        r_loss_factor: Factor that multiplies the loss factor of the
                       reconstruction loss.
        Beta: Beta-VAE parameter that multiplies the KL-Divergence in order to
              disentangle the latent space of the model.
              
        """
        
        self.learning_rate = learning_rate

        ### COMPILATION
        def vae_r_loss(y_true, y_pred):
            r_loss = K.mean(K.square(y_true - y_pred), axis = -1)
            return r_loss_factor * r_loss

        def vae_kl_loss(y_true, y_pred):
            kl_loss =  -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis = 1)
            return kl_loss

        def vae_loss(y_true, y_pred):
            r_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(y_true, y_pred)
            return  r_loss + Beta*kl_loss

        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss = vae_loss,  metrics = [vae_r_loss, vae_kl_loss])

    def train(self, x_train, y_train, batch_size, epochs, run_folder, print_every_n_batches = 100, initial_epoch = 0, lr_decay = 1):
        
        # custom_callback = CustomCallback(run_folder, print_every_n_batches, initial_epoch, self)
        # lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint_filepath=os.path.join(run_folder, "weights/weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = ModelCheckpoint(checkpoint_filepath, save_weights_only = True, verbose=0)
        checkpoint2 = ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'), save_weights_only = True, verbose=0)

        callbacks_list = [checkpoint1, checkpoint2]

        self.model.fit(     
            x_train
            , y_train
            , batch_size = batch_size
            , shuffle = True
            , epochs = epochs
            , initial_epoch = initial_epoch
            , callbacks = callbacks_list
        )


    def save(self,filename):
        '''
            saves the pretrained weights. Saving and loading a keras model 
            after pretraining is better done directly to the self.autoencoder
            object using the keras fnctions save() and load_model()
        '''

        if self.pretrained == True:
            for i in range(self.num_hidden_layers):
                weights = {"W":self.W[i],'a':self.a[i],'b':self.b[i]}
                RBM.save_weights(weights,filename+"_"+str(i))
        else: 
            print("No pretrained weights to save.")

        return    
