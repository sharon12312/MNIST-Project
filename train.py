import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

MODEL_PATH = './Models/'

# read the data and labels as ont-hot vectors
# one-hot means a sparse vector for every observation where only
# the class label is 1, and every other class is 0.
# more info here:
mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)

# mnist is now a DataSet with accessors for:
# 'train', 'test', and 'validation'.
# within each, we can access:
# images, labels, and num_examples
# print(mnist.train.num_examples, mnist.test.num_examples, mnist.validation.num_examples)

# the images are stored as:
# n_observations x n_features tensor (n-dim array)
# the labels are stored as n_observations x n_labels,
# where each observation is a one-hot vector.
# print(mnist.train.images.shape, mnist.train.labels.shape)

# the range of the values of the images is from 0-1
# print(np.min(mnist.train.images), np.max(mnist.train.images))

# we can visualize any one of the images by reshaping it to a 28x28 image
#plt.imshow(np.reshape(mnist.train.images[100, :], (28, 28)), cmap='gray')

# Saving Model Function
def save_model(sess, model_path):
    if model_path is not None:
        print('Saving my model..')

        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess, model_path)

# Messages Console
print('Staring create a model for MNIST..');

# Model Parameters
n_input = 784
n_hidden_1 = 256
n_hidden_2 = 64
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

# prediction and actual using the argmax as the predicted label
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y_true, 1))

# And now we can look at the mean of our network's correct guesses
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Cost Formula
cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Y, labels=y_true))

eta = 1e-4
optimizer = tf.train.AdamOptimizer(eta).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Messages Console
print('Starting to train...')

# Starting Train
batch_size = 50
n_epochs = 1000
l_loss = list()

for epoch_i in range(n_epochs):
    for batch_i in range(0, mnist.train.num_examples, batch_size):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={
            net_input: batch_xs,
            y_true: batch_ys
         })
    loss = sess.run(accuracy, feed_dict={
                       net_input: mnist.validation.images,
                       y_true: mnist.validation.labels})
    print('Validation accuracy for epoch {} is: {}'.format(epoch_i + 1, loss))
    l_loss.append(loss)

# Saving Model
save_model(sess, MODEL_PATH)