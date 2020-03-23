########################################################################
#                                                                      #
# Based on this repository: https://github.com/meownoid/tensorfow-rbm/ #
#                                                                      #
########################################################################


from __future__ import print_function

import tensorflow as tf
import numpy as np

from tqdm import trange

def sample_bernoulli(probs):
    return tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(input=probs))))


def sample_gaussian(x, sigma):
    return x + tf.random.normal(tf.shape(input=x), mean=0.0, stddev=sigma, dtype=tf.float32)


def xavier_init(fan_in, fan_out, constant=1.): 
    """ Xavier initialization of network weights
    https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow    
    
    Arguments:
        fan_in {int} -- [description]
        fan_out {int} -- [description]
    
    Keyword Arguments:
        constant {float} -- Constant (default: {1.})
    
    Returns:
        Tensorflow Tensor -- Xavier Weights.

    """
    
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random.uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

class RBM_base:
    def __init__(self, n_visible, n_hidden, learning_rate=0.01, 
                momentum=0.95, xavier_const=1.0, err_function='mse',
                device='gpu', use_tqdm=False):

        if not 0.0 <= momentum <= 1.0:
            raise ValueError('momentum should be in range [0, 1]')

        if err_function not in {'mse', 'cosine'}:
            raise ValueError('err_function should be either \'mse\' or \'cosine\'')

        self._use_tqdm = use_tqdm        
        self.device = device

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.n_visible])
        self.y = tf.compat.v1.placeholder(tf.float32, [None, self.n_hidden])

        self.w = tf.Variable(xavier_init(self.n_visible, self.n_hidden), dtype=tf.float32, name='W')
        self.visible_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32, name='Visible_bias')
        self.hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32, name='Hidden_bias')

        self.delta_w = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), dtype=tf.float32, name='delta_W')
        self.delta_visible_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32, name='delta_Visible_Bias')
        self.delta_hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32, name='delta_Hidden_Bias')

        #No predifened weights

        self.update_weights = None
        self.update_deltas = None
        self.compute_hidden = None
        self.compute_visible = None
        self.compute_visible_from_hidden = None

        self._initialize_vars()

        #Check if the weights are defined
        assert self.update_weights is not None
        assert self.update_deltas is not None
        assert self.compute_hidden is not None
        assert self.compute_visible is not None
        assert self.compute_visible_from_hidden is not None


        # Add This to the GPU

        if err_function == 'cosine':
            x1_norm = tf.nn.l2_normalize(self.x, 1)
            x2_norm = tf.nn.l2_normalize(self.compute_visible, 1)
            cos_val = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=tf.mul(x1_norm, x2_norm), axis=1))
            self.compute_err = tf.acos(cos_val) / tf.constant(np.pi)
        
        else:
            self.compute_err = tf.reduce_mean(input_tensor=tf.square(self.x - self.compute_visible))

        init = tf.compat.v1.global_variables_initializer()
        
        assert self.device in ('gpu','cpu')

        if self.device == 'gpu':
            self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

        if self.device == 'cpu':
            self.sess = tf.compat.v1.Session()
        
        self.sess.run(init)
    
    def get_err(self, batch_x):
        return self.sess.run(self.compute_err, feed_dict={self.x: batch_x})

    def transform(self, batch_x):
        return self.sess.run(self.compute_hidden, feed_dict={self.x: batch_x})

    def transform_inv(self, batch_y):
        return self.sess.run(self.compute_visible_from_hidden, feed_dict={self.y: batch_y})

    def reconstruct(self, batch_x):
        return self.sess.run(self.compute_visible, feed_dict={self.x: batch_x})

    def partial_fit(self, batch_x):
        self.sess.run(self.update_weights + self.update_deltas, feed_dict={self.x: batch_x})

    def train(self,
            data_x,
            n_epoches=10,
            batch_size=10,
            shuffle=True,
            verbose=True):
        
        assert n_epoches > 0

        n_data = data_x.shape[0]

        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        
        else:
            n_batches = 1        

        if shuffle:
            data_x_cpy = data_x.copy()
            inds = np.arange(n_data)
        
        
        else:
            data_x_cpy = data_x

        errs = []

        for e in range(n_epoches):
            
            if verbose and not self._use_tqdm:
                print(f'Epoch: {e:d}')

            epoch_errs = np.zeros((n_batches,))
            epoch_errs_ptr = 0

            if shuffle:
                np.random.shuffle(inds)
                data_x_cpy = data_x_cpy[inds]

            r_batches = trange(n_batches)            
            
            for b in r_batches:
                batch_x = data_x_cpy[b * batch_size:(b + 1) * batch_size]
                self.partial_fit(batch_x)
                batch_err = self.get_err(batch_x)
                epoch_errs[epoch_errs_ptr] = batch_err
                epoch_errs_ptr += 1

            
            if verbose:
                err_mean = epoch_errs.mean()
                print('Train error: {:.4f}'.format(err_mean))
                print('')
                

            errs.append(epoch_errs.mean())

        return errs

    def get_weights(self):
        
        return self.sess.run(self.w),\
            self.sess.run(self.visible_bias),\
            self.sess.run(self.hidden_bias)

    def save_weights(self, filename, name):
        
        saver = tf.compat.v1.train.Saver({name + '_w': self.w,
                                name + '_v': self.visible_bias,
                                name + '_h': self.hidden_bias})
        return saver.save(self.sess, filename)

    def set_weights(self, w, visible_bias, hidden_bias):
        
        self.sess.run(self.w.assign(w))
        self.sess.run(self.visible_bias.assign(visible_bias))
        self.sess.run(self.hidden_bias.assign(hidden_bias))

    def load_weights(self, filename, name):
                
        saver = tf.compat.v1.train.Saver({name + '_w': self.w,
                                name + '_v': self.visible_bias,
                                name + '_h': self.hidden_bias})
        saver.restore(self.sess, filename)


class RBM(RBM_base):
    
    def __init__(self, n_visible, n_hidden, learning_rate=0.01, momentum=0.95, xavier_const=1.0, err_function='mse', device='gpu', use_tqdm=False):
        super().__init__(n_visible, 
                         n_hidden, 
                         learning_rate=learning_rate, 
                         momentum=momentum, 
                         xavier_const=xavier_const, 
                         err_function=err_function, 
                         device=device, 
                         use_tqdm=use_tqdm)

    def _initialize_vars(self):

        hidden_p = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        visible_recon_p = tf.nn.sigmoid(tf.matmul(sample_bernoulli(hidden_p), tf.transpose(a=self.w)) + self.visible_bias)
        hidden_recon_p = tf.nn.sigmoid(tf.matmul(visible_recon_p, self.w) + self.hidden_bias)

        positive_grad = tf.matmul(tf.transpose(a=self.x), hidden_p)
        negative_grad = tf.matmul(tf.transpose(a=visible_recon_p), hidden_recon_p)

        def f(x_old, x_new):
            return self.momentum * x_old +\
                   self.learning_rate * x_new * (1 - self.momentum) / tf.cast(tf.shape(input=x_new)[0], dtype=tf.float32)

        delta_w_new = f(self.delta_w, positive_grad - negative_grad)
        delta_visible_bias_new = f(self.delta_visible_bias, tf.reduce_mean(input_tensor=self.x - visible_recon_p, axis=0))
        delta_hidden_bias_new = f(self.delta_hidden_bias, tf.reduce_mean(input_tensor=hidden_p - hidden_recon_p, axis=0))

        update_delta_w = self.delta_w.assign(delta_w_new)
        update_delta_visible_bias = self.delta_visible_bias.assign(delta_visible_bias_new)
        update_delta_hidden_bias = self.delta_hidden_bias.assign(delta_hidden_bias_new)

        update_w = self.w.assign(self.w + delta_w_new)
        update_visible_bias = self.visible_bias.assign(self.visible_bias + delta_visible_bias_new)
        update_hidden_bias = self.hidden_bias.assign(self.hidden_bias + delta_hidden_bias_new)

        self.update_deltas = [update_delta_w, update_delta_visible_bias, update_delta_hidden_bias]
        self.update_weights = [update_w, update_visible_bias, update_hidden_bias]

        self.compute_hidden = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        self.compute_visible = tf.nn.sigmoid(tf.matmul(self.compute_hidden, tf.transpose(a=self.w)) + self.visible_bias)
        self.compute_visible_from_hidden = tf.nn.sigmoid(tf.matmul(self.y, tf.transpose(a=self.w)) + self.visible_bias)
        

        hidden_p = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        visible_recon_p = tf.nn.sigmoid(tf.matmul(sample_bernoulli(hidden_p), tf.transpose(a=self.w)) + self.visible_bias)
        hidden_recon_p = tf.nn.sigmoid(tf.matmul(visible_recon_p, self.w) + self.hidden_bias)

        positive_grad = tf.matmul(tf.transpose(a=self.x), hidden_p)
        negative_grad = tf.matmul(tf.transpose(a=visible_recon_p), hidden_recon_p)

        def f(x_old, x_new):
            return self.momentum * x_old +\
                   self.learning_rate * x_new * (1 - self.momentum) / tf.cast(tf.shape(input=x_new)[0], dtype=tf.float32)

        delta_w_new = f(self.delta_w, positive_grad - negative_grad)
        delta_visible_bias_new = f(self.delta_visible_bias, tf.reduce_mean(input_tensor=self.x - visible_recon_p, axis=0))
        delta_hidden_bias_new = f(self.delta_hidden_bias, tf.reduce_mean(input_tensor=hidden_p - hidden_recon_p, axis=0))

        update_delta_w = self.delta_w.assign(delta_w_new)
        update_delta_visible_bias = self.delta_visible_bias.assign(delta_visible_bias_new)
        update_delta_hidden_bias = self.delta_hidden_bias.assign(delta_hidden_bias_new)

        update_w = self.w.assign(self.w + delta_w_new)
        update_visible_bias = self.visible_bias.assign(self.visible_bias + delta_visible_bias_new)
        update_hidden_bias = self.hidden_bias.assign(self.hidden_bias + delta_hidden_bias_new)

        self.update_deltas = [update_delta_w, update_delta_visible_bias, update_delta_hidden_bias]
        self.update_weights = [update_w, update_visible_bias, update_hidden_bias]

        self.compute_hidden = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        self.compute_visible = tf.nn.sigmoid(tf.matmul(self.compute_hidden, tf.transpose(a=self.w)) + self.visible_bias)
        self.compute_visible_from_hidden = tf.nn.sigmoid(tf.matmul(self.y, tf.transpose(a=self.w)) + self.visible_bias)


class RBM_Linear(RBM_base):
    
    def __init__(self, 
                n_visible, 
                n_hidden, 
                sample_visible=True,
                sigma=1.,
                learning_rate=0.01, 
                momentum=0.95, 
                xavier_const=1.0, 
                err_function='mse', 
                device='gpu', 
                use_tqdm=False):

        self.sample_visible = sample_visible
        self.sigma = sigma  
        super().__init__(n_visible, 
                         n_hidden, 
                         learning_rate=learning_rate, 
                         momentum=momentum, 
                         xavier_const=xavier_const, 
                         err_function=err_function, 
                         device=device, 
                         use_tqdm=use_tqdm,
                         )

    def _initialize_vars(self):
        hidden_p = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        visible_recon_p = tf.matmul(sample_bernoulli(hidden_p), tf.transpose(a=self.w)) + self.visible_bias

        if self.sample_visible:
            visible_recon_p = sample_gaussian(visible_recon_p, self.sigma)

        hidden_recon_p = tf.nn.sigmoid(tf.matmul(visible_recon_p, self.w) + self.hidden_bias)

        positive_grad = tf.matmul(tf.transpose(a=self.x), hidden_p)
        negative_grad = tf.matmul(tf.transpose(a=visible_recon_p), hidden_recon_p)

        def f(x_old, x_new):
            return self.momentum * x_old +\
                   self.learning_rate * x_new * (1 - self.momentum) / tf.cast(tf.shape(input=x_new)[0], dtype=tf.float32)

        delta_w_new = f(self.delta_w, positive_grad - negative_grad)
        delta_visible_bias_new = f(self.delta_visible_bias, tf.reduce_mean(input_tensor=self.x - visible_recon_p, axis=0))
        delta_hidden_bias_new = f(self.delta_hidden_bias, tf.reduce_mean(input_tensor=hidden_p - hidden_recon_p, axis=0))

        update_delta_w = self.delta_w.assign(delta_w_new)
        update_delta_visible_bias = self.delta_visible_bias.assign(delta_visible_bias_new)
        update_delta_hidden_bias = self.delta_hidden_bias.assign(delta_hidden_bias_new)

        update_w = self.w.assign(self.w + delta_w_new)
        update_visible_bias = self.visible_bias.assign(self.visible_bias + delta_visible_bias_new)
        update_hidden_bias = self.hidden_bias.assign(self.hidden_bias + delta_hidden_bias_new)

        self.update_deltas = [update_delta_w, update_delta_visible_bias, update_delta_hidden_bias]
        self.update_weights = [update_w, update_visible_bias, update_hidden_bias]

        self.compute_hidden = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        self.compute_visible = tf.matmul(self.compute_hidden, tf.transpose(a=self.w)) + self.visible_bias
        self.compute_visible_from_hidden = tf.matmul(self.y, tf.transpose(a=self.w)) + self.visible_bias