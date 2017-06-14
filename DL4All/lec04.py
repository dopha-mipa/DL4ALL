import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Multi variable linear Regression
# mid term, quiz 점수를 이용해서 기말 성적을 예측해보자
def prac01():
  x1_data = [73., 93., 89., 96., 73.]
  x2_data = [80., 88., 91., 98., 66.]
  x3_data = [75., 93., 90., 100., 70.]
  y_data = [152., 185., 180., 196., 142.] # 실제 기말 점수

  # placeholders for a tensor that will be always fed.
  x1 = tf.placeholder(tf.float32)
  x2 = tf.placeholder(tf.float32)
  x3 = tf.placeholder(tf.float32)
  Y = tf.placeholder(tf.float32)

  w1 = tf.Variable(tf.random_normal([1]), name="weight1")
  w2 = tf.Variable(tf.random_normal([1]), name="weight2")
  w3 = tf.Variable(tf.random_normal([1]), name="weight3")
  b = tf.Variable(tf.random_normal([1]), name="bias")

  hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

  cost = tf.reduce_mean(tf.square(hypothesis - Y))
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
  train = optimizer.minimize(cost)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
          feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
    if step % 10 == 0:
      print(step, "Cost :", cost_val, "\nPrediction\n", hy_val)

def prac02():
  x_data = [[73., 80., 75.],
            [93., 88., 93.],
            [89., 91., 90.],
            [96., 98., 100.],
            [73., 66., 70.]]
  y_data = [[152.], [185.], [180.], [196.], [142.]] # 실제 기말 점수

  # placeholders for a tensor that will be always fed.
  # shape[#data, #properties]
  X = tf.placeholder(tf.float32, shape=[None, 3])
  Y = tf.placeholder(tf.float32, shape=[None, 1])

  W = tf.Variable(tf.random_normal([3, 1]), name="weight")
  b = tf.Variable(tf.random_normal([1]), name="bias")

  hypothesis = tf.matmul(X, W) + b

  cost = tf.reduce_mean(tf.square(hypothesis - Y))
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
  train = optimizer.minimize(cost)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
          feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
      print(step, "Cost :", cost_val, "\nPrediction\n", hy_val)

# with csv! * warning : UTF-8 csv는 읽을수 없어요 흙흙
def prac03():
  # Exam01, Exam02, Exam03, Final
  xy = np.loadtxt("lec04_dataset.csv", delimiter=',', dtype=np.float32)
  x_data = xy[:, 0:-1] # -2까지 들어간댜
  y_data = xy[:, [-1]] # -1만 땋!

  # print(x_data.shape, "\n", x_data, len(x_data))
  # print(y_data.shape, "\n", y_data)

  # placeholders for a tensor that will be always fed.
  # shape[#data, #properties]
  X = tf.placeholder(tf.float32, shape=[None, 3])
  Y = tf.placeholder(tf.float32, shape=[None, 1])

  W = tf.Variable(tf.random_normal([3, 1]), name="weight")
  b = tf.Variable(tf.random_normal([1]), name="bias")

  hypothesis = tf.matmul(X, W) + b

  cost = tf.reduce_mean(tf.square(hypothesis - Y))
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
  train = optimizer.minimize(cost)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
          feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
      print(step, "Cost :", cost_val, "\nPrediction\n", hy_val)

  # Ask my score!
  print("Your score will be ", sess.run(hypothesis, 
        feed_dict={X: [[100., 70., 101.]]}))
  print("Other scores will be ", sess.run(hypothesis,
        feed_dict={X: [[60., 70., 110.], [90., 100., 80]]}))

def seaViolin():
  # 요건 희경이가 알려준 violin plot! 재미로
  # 여러 개를 그리려면 "get"이 있는 데이터형이어야 하나보다..
  x_data = pd.read_csv("lec04_dataset.csv")
  sns.violinplot(data=x_data, palette="Spectral")
  sns.plt.show()

def prac04():
  # TensorFlow.Queue Runners - 여러개의 파일을 불러오는 법에 관한 예제
  filename_queue = tf.train.string_input_producer(
        ["lec04_dataset.csv"], shuffle=False, name="filename_queue")
  reader = tf.TextLineReader()
  key, value = reader.read(filename_queue)

  # 어떤 형태의 데이터 타입인지 미리 지정.
  record_defaults = [[0.], [0.], [0.], [0.]]
  xy = tf.decode_csv(value, record_defaults=record_defaults)

  # collect batches of csv in - batch는 데이터를 읽어오는 일종의 펌프 ~
  train_x_batch, train_y_batch = \
  tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10) # 한번에 10개씩 가져와라

  '''
  # shuffle batch code
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  example_batch, label_batch = tf.train.shuffle_batch(
      [example, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  '''

  X = tf.placeholder(tf.float32, shape=[None, 3])
  Y = tf.placeholder(tf.float32, shape=[None, 1])

  W = tf.Variable(tf.random_normal([3, 1]), name="weight")
  b = tf.Variable(tf.random_normal([1]), name="bias")

  hypothesis = tf.matmul(X, W) + b

  cost = tf.reduce_mean(tf.square(hypothesis - Y))
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
  train = optimizer.minimize(cost)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # Start populating the filename queue
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    # data를 읽어온 뒤 feed_date로 넘겨줍니다
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
          feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
      print(step, "Cost :", cost_val, "\nPrediction\n", hy_val)

  coord.request_stop()
  coord.join(threads)

if __name__ == "__main__":
  # prac01()
  # prac02()
  # prac03()
  # seaViolin()
  prac04()