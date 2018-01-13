import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter
import os
import tensorflow.examples.tutorials.mnist.input_data as input_data

# Directory settings
MODEL_PATH = './Models/'
PATH = './Images'

mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)
n_input = 784
n_hidden_1 = 128
n_hidden_2 = 32
n_output = 10
net_input = tf.placeholder(tf.float32, [None, n_input])
y_true = tf.placeholder(tf.float32, [None, 10])

# First Level
W1 = tf.Variable(tf.truncated_normal([784, n_hidden_1], stddev=0.1))
b1 = tf.Variable(tf.zeros([n_hidden_1]))

# Second Level
W2 = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1))
b2 = tf.Variable(tf.zeros([n_hidden_2]))

# Second Level
W3 = tf.Variable(tf.truncated_normal([n_hidden_2, n_output], stddev=0.1))
b3 = tf.Variable(tf.zeros([n_output]))

# Set Model
XX = tf.reshape(net_input, [-1, 784])
Y1 = tf.nn.relu(tf.matmul(XX, W1) + b1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)
Ylogits = tf.matmul(Y2, W3) + b3
Y = tf.nn.softmax(Ylogits)

## Check Accuracy
def check_accuracy(sess, feed_dict):
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_rates = sess.run(accuracy, feed_dict=feed_dict)

    return accuracy_rates

# Accuracy Test
def accuracy_of_testset(sess):
    print('Calculating accuracy of test set..')

    X = mnist.test.images.reshape([-1, 784])
    Y = mnist.test.labels
    feed_dict = {
        net_input: X,
        y_true: Y
    }

    accuracy = check_accuracy(sess, feed_dict)
    print('Accuracy of test set: %f' % accuracy)

# Load Image and Reshape
def load_image(filename):
    img = Image.open(filename).convert('L')

    # resize to 28x28
    img = img.resize((28, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)

    # normalization : 255 RGB -> 0, 1
    data = [(255 - x) * 1.0 / 255.0 for x in list(img.getdata())]

    # reshape -> [-1, 28, 28, 1]
    return np.reshape(data, [-1, 784]).tolist()

# Classify Image
def classify(sess, data):
    feed_dict = {
        net_input: data
    }
    number = sess.run(tf.argmax(Y, 1), feed_dict)[0]
    accuracy = sess.run(tf.nn.softmax(Y), feed_dict)[0]

    return number, accuracy[number]

# Predict Image
def predict(sess, filename):
    data = load_image(filename)
    number, accuracy = classify(sess, data)
    print('%d is %s, accuracy: %f' % (number, os.path.basename(filename), accuracy))

def predict_images(sess, path):
    print('\nPredicting Images...')
    for root, dirs, files in os.walk(path):
        for file in files:
            predict(sess, PATH + '/' + file)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, MODEL_PATH)

accuracy_of_testset(sess)

# Predict Image
predict_images(sess, PATH)
