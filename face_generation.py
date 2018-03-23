
data_dir = './data'

# FloydHub - Use with data ID "R5KrjnANiKVhLWAkpXhNBe"
#data_dir = '/input'

# GETTING DATA
import helper

helper.download_extract('mnist', data_dir)
helper.download_extract('celeba', data_dir)

show_n_images = 25

# EXPLORE THE DATA

# mnist
%matplotlib inline
import os
from glob import glob
from matplotlib import pyplot

mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'mnist/*.jpg'))[:show_n_images], 28, 28, 'L')
pyplot.imshow(helper.images_square_grid(mnist_images, 'L'), cmap='gray')

# CelebA
show_n_images = 25

mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'img_align_celeba/*.jpg'))[:show_n_images], 28, 28, 'RGB')
pyplot.imshow(helper.images_square_grid(mnist_images, 'RGB'))

# Preprocess the data

# Building the neural network
# model inputs, generator, discriminator, model_loss, model_opt, train

# check version of tensorflow and access to GPU
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# INPUTS
import problem_unittests as tests

def model_inputs(image_width, image_height, image_channels, z_dim):
    """
    Create the model inputs
    :param image_width: The input image width
    :param image_height: The input image height
    :param image_channels: The number of image channels
    :param z_dim: The dimension of Z
    :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
    """
    input_real = tf.placeholder(tf.float32, (None, image_height, image_width, image_channels), name = 'input_real')
    input_z = tf.placeholder(tf.float32, (None, z_dim), name = 'input_z')
    learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')

    return input_real, input_z, learning_rate

tests.test_model_inputs(model_inputs)

def generator(z, out_channel_dim, is_train = True):
    with tf.variable_scope('generator', reuse = not is_train):
        
        alpha = 0.2
        
        h1 = tf.layers.dense(z, 7*7*256)
        
        h2 = tf.reshape(h1, (-1, 7, 7, 256))
        h2 = tf.layers.batch_normalization(h2, training = is_train)
        h2 = tf.maximum(alpha*h2, h2)
        
        h3 = tf.layers.conv2d_transpose(h2, 128, 5, strides = 1, padding = 'same')
        h3 = tf.layers.batch_normalization(h3, training = is_train)
        h3 = tf.maximum(alpha*h3, h3)
        
        h4 = tf.layers.conv2d_transpose(h3, 64, 5, strides = 2, padding = 'same')
        h4 = tf.layers.batch_normalization(h4, training = is_train)
        h4 = tf.maximum(alpha*h4, h4)
        
        logits = tf.layers.conv2d_transpose(h4, out_channel_dim, 5, strides=2, padding='same')
        
        output = tf.tanh(logits)
    
        return output
    
tests.test_generator(generator, tf)

def discriminator(images, reuse = False):
    with tf.variable_scope('discriminator', reuse = reuse):
        
        alpha = 0.2 
        
        h1 = tf.layers.conv2d(images, 64, 5, strides = 2, padding = 'same')
        h1 = tf.maximum(alpha*h1, h1)
        
        h2 = tf.layers.conv2d(h1, 128, 5, strides = 1, padding ='same')
        h2 = tf.layers.batch_normalization(h2, training = True)
        h2 = tf.maximum(alpha*h2, h2)
        
        h3 = tf.layers.conv2d(h2, 256, 5, strides = 2, padding = 'same')
        h3 = tf.layers.batch_normalization(h3, training = True)
        h3 = tf.maximum(alpha*h3, h3)
        
        flatten = tf.reshape(h3, (-1, 7*7*256))
        logits = tf.layers.dense(flatten, 1)
        output = tf.sigmoid(logits)
        
        return output, logits

tests.test_discriminator(discriminator, tf)

def model_loss(input_real, input_z, out_channel_dim):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    # calculate both the lossses simultaneously 
    smooth = 0.1
    # calculate both the lossses simultaneously 
    g_model = generator(input_z, out_channel_dim, is_train = True)
    dout_real, dlogits_real = discriminator(input_real)
    dout_fake, dlogits_fake = discriminator(g_model, reuse = True)

    # generator loss 
    gloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = dlogits_fake, labels = tf.ones_like(dout_fake)))

    # discriminator loss 
    dloss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = dlogits_real, labels = tf.ones_like(dout_real)* (1 - smooth)))
    dloss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = dlogits_fake, labels = tf.zeros_like(dout_fake)))
    
    dloss = dloss_real + dloss_fake

    return dloss, gloss 
    
tests.test_model_loss(model_loss)

def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)): 
        
        d_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

        return d_train_opt, g_train_opt

tests.test_model_opt(model_opt, tf)

import numpy as np

def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):
    """
    Show example output for the generator
    :param sess: TensorFlow session
    :param n_images: Number of Images to display
    :param input_z: Input Z Tensor
    :param out_channel_dim: The number of channels in the output image
    :param image_mode: The mode to use for images ("RGB" or "L")
    """
    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, False),
        feed_dict={input_z: example_z})

    images_grid = helper.images_square_grid(samples, image_mode)
    pyplot.imshow(images_grid, cmap=cmap)
    pyplot.show()

def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
    """
    Train the GAN
    :param epoch_count: Number of epochs
    :param batch_size: Batch Size
    :param z_dim: Z dimension
    :param learning_rate: Learning Rate
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :param get_batches: Function to get batches
    :param data_shape: Shape of the data
    :param data_image_mode: The image mode to use for images ("RGB" or "L")
    """
    # TODO: Build Model
    #tf.reset_default_graph()
    input_real, input_fake, lr = model_inputs(data_shape[1], data_shape[2], data_shape[3], z_dim)
    dloss, gloss = model_loss(input_real, input_fake, data_shape[3])
    d_train_opt, g_train_opt = model_opt(dloss, gloss, learning_rate, beta1)

    print_every = 5
    show_every = 2
    n_images = 25
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            steps = 0
            for batch_images in get_batches(batch_size):
                steps += 1
                batch_z = np.random.uniform(-1, 1, size= (batch_size, z_dim))
                batch_images = batch_images * 2.0
                _ = sess.run(d_train_opt, feed_dict={input_real: batch_images, input_fake: batch_z, lr: learning_rate})
                _ = sess.run(g_train_opt, feed_dict={input_real: batch_images, input_fake: batch_z, lr: learning_rate})

                if steps % print_every == 0:
                    dtrainloss = dloss.eval({input_real: batch_images, input_fake: batch_z, lr: learning_rate})
                    gtrainloss = gloss.eval({input_fake: batch_z})
                    print("Epoch {}/{}....".format(epoch_i + 1, epoch_count), "Discriminator Loss: {:.4f}....".format(dtrainloss), "Generator Loss: {:.4f}".format(gtrainloss))
                
                if steps % 100 == 0:
                    show_generator_output(sess, n_images, input_fake, data_shape[3], data_image_mode)
 

batch_size = 32
z_dim = 100
learning_rate = 0000.2
beta1 = 0.5


epochs = 2

mnist_dataset = helper.Dataset('mnist', glob(os.path.join(data_dir, 'mnist/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, mnist_dataset.get_batches,
          mnist_dataset.shape, mnist_dataset.image_mode)


batch_size = 32
z_dim = 100
learning_rate = 0000.2
beta1 = 0.5


epochs = 1

celeba_dataset = helper.Dataset('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches,
          celeba_dataset.shape, celeba_dataset.image_mode)

# ..................................... PROJECT COMPLETED !! .............................................
