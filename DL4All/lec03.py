import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt

def prac01():
  # pyplot을 이용해서 cost의 그래프 그려보기
  X = [1, 2, 3]
  Y = [1, 2, 3]

  W = tf.placeholder(tf.float32)
  # Our hypothesis for linear model X * W
  hypothesis = X * W

  # 여기선 optimizer를 쓰지 않았습니다
  cost = tf.reduce_mean(tf.square(hypothesis - Y))
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # for graph
  W_val = []
  cost_val = []
  for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

  # Show the cost function
  plt.plot(W_val, cost_val)
  plt.show()

def prac02():
  x_data = [1, 2, 3]
  y_data = [1, 2, 3]

  W = tf.Variable(tf.random_normal([1]), name="weight")
  X = tf.placeholder(tf.float32)
  Y = tf.placeholder(tf.float32)

  # Our hypothesis for linear model X * W
  hypothesis = X * W

  # cost / loss function
  cost = tf.reduce_sum(tf.square(hypothesis - Y))

  # Minimize : Gradient Descent using derivative: W -= learning rate * derivative
  learning_rate = 0.1
  gradient = tf.reduce_mean((W * X - Y) * X) # cost를 미분한 기울기!
  descent = W - learning_rate * gradient
  update = W.assign(descent) # 바로 equal로 assign을 할 수가 없습니당

  '''
  # 요 윗부분을 대신할 수 있는게 바로 lec02에서 했던 
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  train = optimizer.minimize(cost)
  '''

  #launch the graph in a session
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

def prac03():
  X = [1, 2, 3]
  Y = [1, 2, 3]

  W = tf.Variable(-3.0)
  # W = tf.Variable(5.0)

  # Our hypothesis for linear model X * W
  hypothesis = X * W
  # cost / loss function
  cost = tf.reduce_mean(tf.square(hypothesis - Y))

  # Minimize : Gradient Descent magic
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
  train = optimizer.minimize(cost)

  #launch the graph in a session
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  for step in range(100):
    print(step, sess.run(W))
    sess.run(train)

'''
# Optional : compute_gradient and apply_gradient
# gradient 값에 조정을 가하고 싶을때 
'''
def prac04():
  X = [1, 2, 3]
  Y = [1, 2, 3]

  # Set wron model weights
  W = tf.Variable(5.0)

  # Our hypothesis for linear model X * W
  hypothesis = X * W

  # Manual gradient : 수식적으로 계산된 gradient
  gradient = tf.reduce_mean((W * X - Y) * X) * 2
  # cost / loss function
  cost = tf.reduce_mean(tf.square(hypothesis - Y))

  # Minimize : Gradient Descent magic
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

  # ☆ 기울기를 알려주세요!
  gvs = optimizer.compute_gradients(cost)
  # Apply gradients - 수정해서 apply 할 수 있어요~ 안하면 minimize랑 같다
  apply_gradients = optimizer.apply_gradients(gvs)

  #launch the graph in a session
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)

if __name__ == "__main__":
  # prac01()
  # prac02()
  # prac03()
  prac04()