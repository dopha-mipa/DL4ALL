import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def prac01():
  x_data = [[1, 2],
            [2, 3],
            [3, 1],
            [4, 3],
            [5, 3],
            [6, 2]]
  y_data = [[0], [0], [0], [1], [1], [1]]

  #placeholders for a tensor that will be always fed
  X = tf.placeholder(tf.float32, shape=[None, 2])
  Y = tf.placeholder(tf.float32, shape=[None, 1])

  W = tf.Variable(tf.random_normal([2, 1]), name="weight")
  b = tf.Variable(tf.random_normal([1, 1]), name="bias")

  # Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)) +  b)
  hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

  cost = -1 * tf.reduce_mean(Y * tf.log(hypothesis)
         + (1 - Y) * tf.log(1 - hypothesis))

  train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

  # Accuracy computation
  # True if hypothesis > 0.5 else False - T/F로 결과값을 캐스팅
  predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

  with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
      cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
      if step % 200 == 0:
        print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], 
            feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

def classifying_diabetes():
  xy = np.loadtxt("lec05_diabetes.csv", delimiter=",", dtype=np.float32)
  x_data = xy[:, 0:-1]
  y_data = xy[:, [-1]]

#placeholders for a tensor that will be always fed
  X = tf.placeholder(tf.float32, shape=[None, 8])
  Y = tf.placeholder(tf.float32, shape=[None, 1])

  W = tf.Variable(tf.random_normal([8, 1]), name="weight")
  b = tf.Variable(tf.random_normal([1, 1]), name="bias")

  # Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)) +  b)
  hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

  cost = -1 * tf.reduce_mean(Y * tf.log(hypothesis)
         + (1 - Y) * tf.log(1 - hypothesis))

  train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

  # Accuracy computation
  predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    feed = {X: x_data, Y: y_data}
    for step in range(10001):
      sess.run(train, feed_dict=feed)
      if step % 200 == 0:
        print(step, sess.run(cost, feed_dict=feed))

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict=feed)
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)


if __name__ == "__main__":
  # prac01()
  classifying_diabetes()