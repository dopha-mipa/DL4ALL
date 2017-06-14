import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import random
tf.set_random_seed(777)  # reproducibility
from tensorflow.examples.tutorials.mnist import input_data

def prac01():
  '''
  # Image : 1, 3, 3, 1
  # Filter : 2, 2, 1, 1 (2, 2), color : 1, #filters = 1
  # Stride : 1 * 1
  # Padding : Valid
  '''
  sess = tf.InteractiveSession()
  image = np.array([[[[1], [2], [3]],
                     [[4], [5], [6]],
                     [[7], [8], [9]]]], dtype=np.float32)
  print(image.shape)
  plt.imshow(image.reshape(3, 3), cmap='Greys')
  plt.show()

  print("image.shape ", image.shape)
  weight = tf.constant([[[[1.]], [[1.]]], # filter!
                        [[[1.]], [[1.]]]])
  print("weight.shape", weight.shape)
  conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding="VALID")
  conv2d_img = conv2d.eval()
  print("conv2d_img.shape", conv2d_img.shape)
  conv2d_img = np.swapaxes(conv2d_img, 0, 3)
  for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(2, 2))
    plt.subplot(1, 2, i + 1)
    plt.imshow(one_img.reshape(2, 2), cmap="gray")

  plt.show()

# 한 줄의 padding을 더해 output 크기가 3 * 3이 되도록
def prac02():
  '''
  # Image : 1, 3, 3, 1
  # Filter : 2, 2, 1, 1 (2, 2), color : 1, #filters = 1
  # Stride : 1 * 1
  # Padding : Same
  '''
  sess = tf.InteractiveSession()
  image = np.array([[[[1], [2], [3]],
                     [[4], [5], [6]],
                     [[7], [8], [9]]]], dtype=np.float32)
  print(image.shape)
  # plt.imshow(image.reshape(3, 3), cmap='Greys')
  # plt.show()

  print("image.shape ", image.shape)
  weight = tf.constant([[[[1.]], [[1.]]], # filter!
                        [[[1.]], [[1.]]]])
  print("weight.shape", weight.shape)
  conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding="SAME")
  conv2d_img = conv2d.eval()
  print("conv2d_img.shape", conv2d_img.shape)
  conv2d_img = np.swapaxes(conv2d_img, 0, 3)
  for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3, 3))
    plt.subplot(1, 2, i + 1)
    plt.imshow(one_img.reshape(3, 3), cmap="gray")

  plt.show()

# 3 Filters! (2, 2, 1, 3) - 3 장의 이미지가 나와요
def prac03():
  '''
  # Image : 1, 3, 3, 1
  # Filter : 2, 2, 1, 3 (2, 2), color : 1, #filters = 3
  # Stride : 1 * 1
  # Padding : Same
  '''
  sess = tf.InteractiveSession()
  image = np.array([[[[1], [2], [3]],
                     [[4], [5], [6]],
                     [[7], [8], [9]]]], dtype=np.float32)
  print(image.shape)
  # plt.imshow(image.reshape(3, 3), cmap='Greys')
  # plt.show()

  print("image.shape ", image.shape)
  weight = tf.constant([[[[1., 10., -1.]], [[1., 10., -1.]]], # filter!
                        [[[1., 10., -1.]], [[1., 10., -1.]]]])
  print("weight.shape", weight.shape)
  conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding="SAME")
  conv2d_img = conv2d.eval()
  print("conv2d_img.shape", conv2d_img.shape)
  conv2d_img = np.swapaxes(conv2d_img, 0, 3)
  for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3, 3))
    plt.subplot(1, 3, i + 1)
    plt.imshow(one_img.reshape(3, 3), cmap="gray")

  plt.show()

# Max Pooling - Convolution 이후의 작업이야!
def prac04():
  '''
  # Image : 1, 3, 3, 1
  # Filter : 1, 2, 1, 1 (2, 2), color : 1, #filters = 3
  # Stride : 1 * 1
  # Padding : Same - 입력과 출력의 크기를 같게 해주세요!!!
  '''
  sess = tf.InteractiveSession()
  image = np.array([[[[4], [3]],
                     [[2], [1]]]], dtype=np.float32)
  pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1], # filter size.. 라는데
                        strides=[1, 1, 1, 1], padding="SAME")
  print(pool.shape)
  print(pool.eval)

  '''
  # 2 * 2 이미지가 padding을 통해 3 * 3이 됐다가
  # 2 * 2 필터를 이용해서 2 * 2의 결과값이 나왔어
  '''

def prac05():
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

  img = mnist.train.images[0].reshape(28, 28)
  plt.imshow(img, cmap="gray")
  plt.show()

  sess = tf.InteractiveSession()
  # (28, 28)의 단색 이미지 
  img = img.reshape(-1, 28, 28, 1) # -1 : 몇 개가 들어오는지는 알아서 계싼해!
  # (3, 3), 단색 / 5개의 필터
  W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01))
  # stride : (1, 1)을 가로 세로 두 칸씩 - output은 14 * 14
  conv2d = tf.nn.conv2d(img, W1, strides=[1, 2, 2, 1], padding="SAME")
  # print(conv2d) # 5장의 레이어

  # 단순 출력을 위한 부분
  sess.run(tf.global_variables_initializer())
  conv2d_img = conv2d.eval()
  conv2d_img = np.swapaxes(conv2d_img, 0, 3)
  for i, one_img in enumerate(conv2d_img):
    plt.subplot(1, 5, i + 1)
    plt.imshow(one_img.reshape(14, 14), cmap="gray")
  # plt.show()

  # Max pooling을 이용한 subsampling
  pool = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], 
                        stride=[1, 2, 2, 1], padding="SAME")
  print(pool)
  sess.rum(tf.global_variables_initializer())
  pool_img = pool.eval()
  pool_img = np.swapaxes(pool_img, 0, 3)
  for i, one_img in enumerate(pool_img):
    plt.subplot(1, 5, i + 1)
    plt.imshow(one_img.reshape(7, 7), cmap="gray")
  plt.show()

# 20분이 걸렸다. 수고했어.. 0.9888
def prac06():
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

  # hyper parameters
  learning_rate = 0.001
  training_epochs = 15
  batch_size = 100

  # input place holders
  X = tf.placeholder(tf.float32, [None, 784])
  X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 단색 (black/white)
  Y = tf.placeholder(tf.float32, [None, 10])

  # L1 ImgIn shape=(?, 28, 28, 1) - 3 * 3, 32 filters
  W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
  #    Conv     -> (?, 28, 28, 32)
  #    Pool     -> (?, 14, 14, 32)
  L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
  L1 = tf.nn.relu(L1) # ReLU를 통과시키고 
  # pooling을 할겁니다. stride (2, 2)
  L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                      strides=[1, 2, 2, 1], padding='SAME')
  '''
  Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
  Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
  Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
  '''

  # L2 ImgIn shape=(?, 14, 14, 32) 32는 똑같이 내려오고!
  W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
  #    Conv      ->(?, 14, 14, 64)
  #    Pool      ->(?, 7, 7, 64)
  L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
  L2 = tf.nn.relu(L2)
  L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                      strides=[1, 2, 2, 1], padding='SAME')
  # 쭉~ 펼쳐줄거래! 64개의 conv들을?
  L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])
  '''
  Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
  Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
  Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
  Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32)
  '''

  # 이 3136개짜리를 이제 Fully Connected Layer로 연결할거야!
  # Final FC 7x7x64 inputs -> 10 outputs
  W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10],
                       initializer=tf.contrib.layers.xavier_initializer())
  b = tf.Variable(tf.random_normal([10]))
  logits = tf.matmul(L2_flat, W3) + b

  # define cost/loss & optimizer
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      logits=logits, labels=Y))
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

  # initialize
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # train my model
  print('Learning started. It takes sometime.')
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
  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print('Accuracy:', sess.run(accuracy, feed_dict={
        X: mnist.test.images, Y: mnist.test.labels}))

  # Get one and predict
  r = random.randint(0, mnist.test.num_examples - 1)
  print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
  print("Prediction: ", sess.run(
      tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

  # plt.imshow(mnist.test.images[r:r + 1].
  #           reshape(28, 28), cmap='Greys', interpolation='nearest')
  # plt.show()

# ensemble learning을 위해서라도 클래스화 시키는것이 좋다.
def prac07():
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  # Check out https://www.tensorflow.org/get_started/mnist/beginners for
  # more information about the mnist dataset

  # hyper parameters
  learning_rate = 0.001
  training_epochs = 15
  batch_size = 100


  class Model:

    def __init__(self, sess, name):
      self.sess = sess
      self.name = name
      self._build_net()

    def _build_net(self):
      with tf.variable_scope(self.name):
        # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
        # for testing
        self.keep_prob = tf.placeholder(tf.float32)

        # input place holders
        self.X = tf.placeholder(tf.float32, [None, 784])
        # img 28x28x1 (black/white)
        X_img = tf.reshape(self.X, [-1, 28, 28, 1])
        self.Y = tf.placeholder(tf.float32, [None, 10])

        # L1 ImgIn shape=(?, 28, 28, 1)
        W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
        L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)
        '''
        Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
        Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
        Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
        Tensor("dropout/mul:0", shape=(?, 14, 14, 32), dtype=float32)
        '''

        # L2 ImgIn shape=(?, 14, 14, 32)
        W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
        L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
        '''
        Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
        Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
        Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
        Tensor("dropout_1/mul:0", shape=(?, 7, 7, 64), dtype=float32)
        '''

        # L3 ImgIn shape=(?, 7, 7, 64)
        W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        #    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
        L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
        L3 = tf.nn.relu(L3)
        L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                            1, 2, 2, 1], padding='SAME')
        L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)

        L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])
        '''
        Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
        Tensor("Relu_2:0", shape=(?, 7, 7, 128), dtype=float32)
        Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
        Tensor("dropout_2/mul:0", shape=(?, 4, 4, 128), dtype=float32)
        Tensor("Reshape_1:0", shape=(?, 2048), dtype=float32)
        '''

        # L4 FC 4x4x128 inputs -> 625 outputs
        W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625],
                             initializer=tf.contrib.layers.xavier_initializer())
        b4 = tf.Variable(tf.random_normal([625]))
        L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
        L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)
        '''
        Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
        Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
        '''

        # L5 Final FC 625 inputs -> 10 outputs
        W5 = tf.get_variable("W5", shape=[625, 10],
                             initializer=tf.contrib.layers.xavier_initializer())
        b5 = tf.Variable(tf.random_normal([10]))
        self.logits = tf.matmul(L4, W5) + b5
        '''
        Tensor("add_1:0", shape=(?, 10), dtype=float32)
        '''

      # define cost/loss & optimizer
      self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
          logits=self.logits, labels=self.Y))
      self.optimizer = tf.train.AdamOptimizer(
          learning_rate=learning_rate).minimize(self.cost)

      correct_prediction = tf.equal(
          tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, keep_prop=1.0):
      return self.sess.run(self.logits, feed_dict={self.X: x_test, self.keep_prob: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
      return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})

    def train(self, x_data, y_data, keep_prop=0.7):
      return self.sess.run([self.cost, self.optimizer], feed_dict={
          self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})

  # initialize
  sess = tf.Session()
  m1 = Model(sess, "m1")

  sess.run(tf.global_variables_initializer())

  print('Learning Started!')

  # train my model
  for epoch in range(training_epochs):
      avg_cost = 0
      total_batch = int(mnist.train.num_examples / batch_size)

      for i in range(total_batch):
          batch_xs, batch_ys = mnist.train.next_batch(batch_size)
          c, _ = m1.train(batch_xs, batch_ys)
          avg_cost += c / total_batch

      print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

  print('Learning Finished!')

  # Test model and check accuracy
  print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))

if __name__ == "__main__":
  prac06()