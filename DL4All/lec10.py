import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import random
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)  # reproducibility

# optimizer에 adam이 사용되었다. 권장됨.
def prac01():
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

  # parameters
  learning_rate = 0.001
  training_epochs = 15
  batch_size = 100
  nb_classes = 10

  # input place holders
  X = tf.placeholder(tf.float32, [None, 784])
  Y = tf.placeholder(tf.float32, [None, 10])

  # weights & bias for nn layers
  # lec09_MNIST의 코드와 달라진 점 : ReLU 적용, xavier initializer 사용!
  # 실감하는 초기값의 힘....!!!
  # W1 = tf.Variable(tf.random_normal([784, 256]))
  W1 = tf.get_variable("W1", shape=[784, 256],
        initializer=tf.contrib.layers.xavier_initializer())
  b1 = tf.Variable(tf.random_normal([256]))
  layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)

  W2 = tf.Variable(tf.random_normal([256, 256]))
  b2 = tf.Variable(tf.random_normal([256]))
  layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

  W3 = tf.Variable(tf.random_normal([256, nb_classes]))
  b3 = tf.Variable(tf.random_normal([nb_classes]))

  hypothesis = tf.matmul(layer2, W3) + b3

  # define cost/loss & optimizer
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      logits=hypothesis, labels=Y))
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

  # initialize
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # train my model
  for epoch in range(training_epochs):
      avg_cost = 0
      total_batch = int(mnist.train.num_examples / batch_size)

      for i in range(total_batch):
          batch_xs, batch_ys = mnist.train.next_batch(batch_size)
          feed_dict = {X: batch_xs, Y: batch_ys}
          c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
          avg_cost += c / total_batch

      print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

  print('Learning Finished!')

  # Test model and check accuracy
  correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print('Accuracy:', sess.run(accuracy, feed_dict={
        X: mnist.test.images, Y: mnist.test.labels}))

  # Get one and predict
  r = random.randint(0, mnist.test.num_examples - 1)
  print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
  print("Prediction: ", sess.run(
      tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

  # plt.imshow(mnist.test.images[r:r + 1].
  #           reshape(28, 28), cmap='Greys', interpolation='nearest')
  # plt.show()

# 97% 실화냐.. 그런데 실망이래 prac01() 결과보다 좀 낮거든ㅋㅋㅋㅋㅋ
# 깊어지면서 과적합~~~
# 그래서 Dropout 레이어를 추가!
# 통상적으로 train에서는 0.5 ~ 0.7, testing에서는 1 ! (각 분야 전문가 총집합)
# drop out 이후 cost 실화냐.. 왕 낮네....
def prac02():
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  # Check out https://www.tensorflow.org/get_started/mnist/beginners for
  # more information about the mnist dataset

  # parameters
  learning_rate = 0.001
  training_epochs = 15
  batch_size = 100

  # 뉴비!
  keep_prob = tf.placeholder(tf.float32)

  # input place holders
  X = tf.placeholder(tf.float32, [None, 784])
  Y = tf.placeholder(tf.float32, [None, 10])

  # weights & bias for nn layers
  # http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
  W1 = tf.get_variable("W1", shape=[784, 512],
                       initializer=tf.contrib.layers.xavier_initializer())
  b1 = tf.Variable(tf.random_normal([512]))
  L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
  L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

  W2 = tf.get_variable("W2", shape=[512, 512],
                       initializer=tf.contrib.layers.xavier_initializer())
  b2 = tf.Variable(tf.random_normal([512]))
  L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
  L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

  W3 = tf.get_variable("W3", shape=[512, 512],
                       initializer=tf.contrib.layers.xavier_initializer())
  b3 = tf.Variable(tf.random_normal([512]))
  L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
  L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

  W4 = tf.get_variable("W4", shape=[512, 512],
                       initializer=tf.contrib.layers.xavier_initializer())
  b4 = tf.Variable(tf.random_normal([512]))
  L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)

  W5 = tf.get_variable("W5", shape=[512, 10],
                       initializer=tf.contrib.layers.xavier_initializer())
  b5 = tf.Variable(tf.random_normal([10]))
  hypothesis = tf.matmul(L4, W5) + b5

  # define cost/loss & optimizer
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      logits=hypothesis, labels=Y))
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

  # initialize
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # train my model
  for epoch in range(training_epochs):
      avg_cost = 0
      total_batch = int(mnist.train.num_examples / batch_size)

      for i in range(total_batch):
          batch_xs, batch_ys = mnist.train.next_batch(batch_size)
          # feed_dict = {X: batch_xs, Y: batch_ys}
          feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
          c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
          avg_cost += c / total_batch

      print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

  print('Learning Finished!')

  # Test model and check accuracy
  correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print('Accuracy:', sess.run(accuracy, feed_dict={
        X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

  # Get one and predict
  r = random.randint(0, mnist.test.num_examples - 1)
  print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
  print("Prediction: ", sess.run(
      tf.argmax(hypothesis, 1), 
          feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))

  # plt.imshow(mnist.test.images[r:r + 1].
  #           reshape(28, 28), cmap='Greys', interpolation='nearest')
  # plt.show()

if __name__ == "__main__":
  # prac01()
  prac02()