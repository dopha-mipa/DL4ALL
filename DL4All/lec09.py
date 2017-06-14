import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.set_random_seed(777)  # for reproducibility

'''
# NN for XOR
'''
def prac01():
  # logistic regression
  # train = test 니 혜택을 많이 받는 모델.. ㅇㅅㅇ
  x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
  y_data = [[0], [1], [1], [0]]
  x_data = np.array(x_data, dtype=np.float32)
  y_data = np.array(y_data, dtype=np.float32)

  X = tf.placeholder(tf.float32, [None, 2])
  Y = tf.placeholder(tf.float32, [None, 1])

  '''
  W = tf.Variable(tf.random_normal([2, 1]), name='weight')
  b = tf.Variable(tf.random_normal([1]), name='bias')

  # Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
  # hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
  '''
  '''
  W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
  b1 = tf.Variable(tf.random_normal([2]), name='bias1')
  layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

  W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
  b2 = tf.Variable(tf.random_normal([1]), name='bias2')
  hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)
  '''

  '''
  W1 = tf.Variable(tf.random_normal([2, 10]), name='weight1')
  b1 = tf.Variable(tf.random_normal([10]), name='bias1')
  layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

  W2 = tf.Variable(tf.random_normal([10, 1]), name='weight2')
  b2 = tf.Variable(tf.random_normal([1]), name='bias2')
  hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)
  '''
  # wide tensor
  W1 = tf.Variable(tf.random_normal([2, 10]), name='weight1')
  b1 = tf.Variable(tf.random_normal([10]), name='bias1')
  layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

  W2 = tf.Variable(tf.random_normal([10, 10]), name='weight2')
  b2 = tf.Variable(tf.random_normal([10]), name='bias2')
  layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

  W3 = tf.Variable(tf.random_normal([10, 10]), name='weight3')
  b3 = tf.Variable(tf.random_normal([10]), name='bias3')
  layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

  W4 = tf.Variable(tf.random_normal([10, 1]), name='weight4')
  b4 = tf.Variable(tf.random_normal([1]), name='bias4')
  hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)

  # cost/loss function
  cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                         tf.log(1 - hypothesis))

  train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

  # Accuracy computation
  # True if hypothesis>0.5 else False
  predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

  # Launch graph
  with tf.Session() as sess:
      # Initialize TensorFlow variables
      sess.run(tf.global_variables_initializer())

      for step in range(10001):
          sess.run(train, feed_dict={X: x_data, Y: y_data})
          if step % 100 == 0:
              print(step, sess.run(cost, feed_dict={
                    X: x_data, Y: y_data}), sess.run(W))

      # Accuracy report
      h, c, a = sess.run([hypothesis, predicted, accuracy],
                         feed_dict={X: x_data, Y: y_data})
      print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)

  '''
  Hypothesis:  [[ 0.5]
   [ 0.5]
   [ 0.5]
   [ 0.5]]
  Correct:  [[ 0.]
   [ 0.]
   [ 0.]
   [ 0.]]
  Accuracy:  0.5
  '''

def prac01():
  pass()

if __name__ == "__main__":
  prac01()