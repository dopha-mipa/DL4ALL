import tensorflow as tf
# tensorflow.org/get_started/mnist/beginners
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import random
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.set_random_seed(777)  # for reproducibility

def prac01():
  # x_data, y_data를 드뎌 깃헙에서 가져오기로 맘 먹음
  x_data = [[1, 2, 1],
            [1, 3, 2],
            [1, 3, 4],
            [1, 5, 5],
            [1, 7, 5],
            [1, 2, 5],
            [1, 6, 6],
            [1, 7, 7]]
  y_data = [[0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0]]

  # Evaluation our model using this test dataset
  x_test = [[2, 1, 1],
            [3, 1, 2],
            [3, 3, 4]]
  y_test = [[0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]]

  X = tf.placeholder("float", [None, 3])
  Y = tf.placeholder("float", [None, 3])

  W = tf.Variable(tf.random_normal([3, 3]), name="weight")
  b = tf.Variable(tf.random_normal([3]), name="bias")

  hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
  # Q : Y 자체가 one_hot이라 그런거야? ㅜㅠㅠ
  cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
  # learning_rate=1.5면 큰일 남 하다가 nan nan 밖에 안나와...
  # 1e-10 이면 그냥 안 나옴
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

  # Correct prediction Test model
  prediction = tf.arg_max(hypothesis, 1)
  is_correct = tf.equal(prediction, tf.arg_max(Y, 1))
  accuracy =  tf.reduce_mean(tf.cast(is_correct, tf.float32))

  # Launch graph
  with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    for step in range(201):
      cost_val, W_val, _ = sess.run([cost, W, optimizer], 
            feed_dict={X: x_data, Y: y_data})
      print(step, cost_val, W_val)

    print("Prediction : ", sess.run(prediction, feed_dict={X: x_test}))
    # Calculate the accuracy
    print("Accuracy : ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))


# Normalize inputs!
def prac02():
  xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
                 [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
                 [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
                 [816, 820.958984, 1008100, 815.48999, 819.23999],
                 [819.359985, 823, 1188100, 818.469971, 818.97998],
                 [819, 823, 1198100, 816, 820.450012],
                 [811.700012, 815.25, 1098100, 809.780029, 813.669983],
                 [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

  xy = MinMaxScaler(xy)

  x_data = xy[:, 0:-1]
  y_data = xy[:, [-1]]

  X = tf.placeholder(tf.float32, shape=[None, 4])
  Y = tf.placeholder(tf.float32, shape=[None, 1])

  W = tf.Variable(tf.random_normal([4, 1]), name="weight")
  b = tf.Variable(tf.random_normal([1]), name="bias")

  hypothesis = tf.matmul(X, W) + b
  cost = tf.reduce_mean(tf.square(hypothesis - Y))

  optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
  train = optimizer.minimize(cost)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  for step in range(2001):
    cost_val, hy_val, _ = sess.run(
          [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    print(step, "Cost :", cost_val, "\nPrediction :", hy_val)

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

# ☆ MNIST Dataset!
def prac03():
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

  # 0 ~ 9 : 10 classes
  nb_classes = 10
  # 28 * 28 shape
  X = tf.placeholder(tf.float32, [None, 784])
  Y = tf.placeholder(tf.float32, [None, nb_classes]) # one_hot

  W = tf.Variable(tf.random_normal([784, nb_classes]))
  b = tf.Variable(tf.random_normal([nb_classes]))

  hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
  cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

  # Test model
  is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
  # Calculate accuracy
  accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

  # batch로 잘라서 학습시키자! 
  # epoch : 전체 데이터 셋을 한번 다 학습시키는 것을 epoch라고 한다.
  training_epochs = 15
  batch_size = 100

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(training_epochs):
      avg_cost = 0
      total_batch = int(mnist.train.num_examples / batch_size)

      for i in range(total_batch):
        # Q : data를 100개씩 읽어들이자... 이건 가중치 계속 바꿀테니까 온라인인가..?
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = sess.run([cost, optimizer], 
            feed_dict={X: batch_xs, Y: batch_ys})
        avg_cost += c / total_batch

      print("Epoch:", "%04d" % (epoch + 1),
            "cost =", "{:.9f}".format(avg_cost))
    
    print("Learning finished")

    print("Accuracy :", accuracy.eval(session=sess,
        feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1) # 임의의 수
    # 하나 읽어오고
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

    # plt.imgshow(mnist.test.images[r:r + 1].reshape(28, 28), 
          # cmp="Greys", interpolation="nearest")
    # plt.show()

if __name__ == "__main__":
  # prac01()
  # prac02()
  prac03()