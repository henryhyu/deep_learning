from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# one hot encoding done already on the feature set (x's)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_prediction = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_prediction,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def filter(shape):
    # Outputs random values from a truncated normal distribution with stddev 0.1
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias(shape):
    # Set bias as a constant
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(input, filter):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    # "SAME" Padding is Zero Padding
    return tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME') 

def max_pool_2x2(input):
    # stride [1, x_movement, y_movement, 1]
    # ksize = the size of the pooling kernel
    # 2x2 pooling window taking strides of 2
    return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Placeholders (aka features feeded in, labels coming out)
xs = tf.placeholder(tf.float32, [None, 784])   # input 28x28 image
ys = tf.placeholder(tf.float32, [None, 10])    # output digits 0-9
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1]) # takes the image in flattened, reshapes it to original dimensions
# 28,28,1 because height, width = 28, channel = 1 (because greyscale, not rgb)
# -1 can also be used to infer the shape, not sure what -1 does..., batch size, sample size?
# [-1, 28, 28, 1] -> [batch size dimension, height dimension, width dimension, channel dimension]
# print(x_image.shape)  # [n_samples, 28,28,1]

# Convolutional Layer 1
filter_c1 = filter([5,5, 1, 32]) # 5x5 filter, in_channel = 1 (same as channel dimension as input, out_channel = 32 (using 32 filters)
bias_c1 = bias([32]) # 32 bias terms
activation_c1 = tf.nn.relu(conv2d(x_image, filter_c1) + bias_c1) # output = 28x28x32
pool_c1 = max_pool_2x2(activation_c1) # output size 14x14x32

# Convolutional Layer 2
filter_c2 = filter([5,5, 32, 64]) # 5x5 filter, in_channel = 32, out_channel = 64 (using 64 filters)
bias_c2 = bias([64]) # 64 bias terms
activation_c2 = tf.nn.relu(conv2d(pool_c1, filter_c2) + bias_c2) # output size 14x14x64
pool_2 = max_pool_2x2(activation_c2) # output size 7x7x64

# Fully Connected Layer 1
weight_fc1 = filter([7*7*64, 1024]) # collapse dimensions of pool_2 (7x7x64), and use 1024 neurons in this layer
bias_fc1 = bias([1024]) # 1024 bias terms
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
pool_2_flat = tf.reshape(pool_2, [-1, 7*7*64]) # need to flatten our inputs from the convolutional layer
activation_fc1 = tf.nn.relu(tf.matmul(pool_2_flat, weight_fc1) + bias_fc1)
dropout_fc1 = tf.nn.dropout(activation_fc1, keep_prob)

# Fully Connected Layer 2
weight_fc2 = filter([1024, 10]) # takes in 1024 inputs from previous layer, outputs 10
bias_fc2 = bias([10]) # 10 bias terms
prediction = tf.nn.softmax(tf.matmul(dropout_fc1, weight_fc2) + bias_fc2)

# error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1])) # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# MomentumOptimizer gets stuck at local gradient with learning rate 0.1, momentum 0.9
# AdamOptimizer gets stuck at 1e-9 (or if epsilon is too small)  Is a very small number to prevent any division by zero in the implementation

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # feed_dict feeds data into the placeholders
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print("Accuracy: ", compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))