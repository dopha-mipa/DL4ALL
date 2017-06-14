import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def prac01():
  # data - x = y네
  x_train = [1, 2, 3]
  y_train = [1, 2, 3]

  # Variable을 tensorflow가 자체적으로 변경 가능한 값이다.
  # 학습 과정에서 변경되는 값이다 (!)
  W = tf.Variable(tf.random_normal([1]), name="weight")
  b = tf.Variable(tf.random_normal([1]), name="bias")

  # hypothesis
  hypothesis = x_train * W + b

  # cost(loss) function - 값의 차이를 제곱해서 합의 평균.
  cost = tf.reduce_mean(tf.square(hypothesis - y_train))

  # gradient Descent.. Minimize
  # learning rate = 가중치를 갱신할 크기.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  train = optimizer.minimize(cost)

  # Launch the graph in a session
  sess = tf.Session()
  # Initializes global variables in the graph
  # 선언해야 Variable()을 이용해 선언했던 W, b를 사용할 수 있다. 초기화
  sess.run(tf.global_variables_initializer())

  # Fit the line (linear)
  for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
      print(step, sess.run(cost), sess.run(W), sess.run(b))

def prac02():
  # data를 미리 선언하는 대신 placeholder로!
  # placeholders for a tensor that will be always fed using feed_dict
  # see http://stackoverflow.com/questions/36693740
  X = tf.placeholder(tf.float32)
  Y = tf.placeholder(tf.float32)

  W = tf.Variable(tf.random_normal([1]), name="weight")
  b = tf.Variable(tf.random_normal([1]), name="bias")

  hypothesis = X * W + b

  cost = tf.reduce_mean(tf.square(hypothesis - Y))

  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  train = optimizer.minimize(cost)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # Fit the line (linear)
  for step in range(2001):
    # sess.run(train) 코드가 어떻게 달라졌나 확인해보는 부분!
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train], feed_dict={
            X: [1, 2, 3, 4, 5],
            Y: [1.1, 2.1, 3.1, 4.1, 5.1]})
    if step % 20 == 0:
      print(step, cost_val, W_val, b_val)

  # 학습이 끝났다! 모델이 예측을 잘 하나 시험해 보자.
  # Testing our model - hypothesis!
  print(sess.run(hypothesis, feed_dict={X: [5]}))
  print(sess.run(hypothesis, feed_dict={X: [2.5]}))
  print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))

if __name__ == "__main__":
  prac02()
  # prac01()