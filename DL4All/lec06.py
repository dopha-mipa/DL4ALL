import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def prac01():
  x_data = [[1, 2, 1, 1],
            [2, 1, 3, 2],
            [3, 1, 3, 4],
            [4, 1, 5, 5],
            [1, 7, 5, 5],
            [1, 2, 5, 6],
            [1, 6, 6, 6],
            [1, 7, 7, 7]]
  y_data = [[0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0]]

  X = tf.placeholder("float", [None, 4])
  Y = tf.placeholder("float", [None, 3])

  nb_classes = 3
  W = tf.Variable(tf.random_normal([4, nb_classes]), name="weight")
  b = tf.Variable(tf.random_normal([nb_classes]), name="bias")


  # hypothesis : 예측값 - softmax 이용해 확률로 변환
  # softmax = exp(logits) / reduce_sum (exp(logits), dim)
  hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
  cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

  # launch graph
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
      sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
      if step % 200 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

# lab06-2 : cross_entropy, ont_hot, *reshape*
def prac02():
  xy = np.loadtxt("lec06_zoo.csv", delimiter=",", dtype=np.float32)
  x_data = xy[:, 0:-1]
  y_data = xy[:, [-1]]
  nb_classes = 7 # 7종류의 동물들

  X = tf.placeholder(tf.float32, [None, 16])
  Y = tf.placeholder(tf.int32, [None, 1])

  # ★ 아랫줄 실행 후의 모양 : one hot shape = (?, 1, 7)
  Y_one_hot = tf.one_hot(Y, nb_classes)
  # one_hot 시에 dimension이 늘어나요. -> (?, 7)로 바꿔주기
  Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

  W = tf.Variable(tf.random_normal([16, nb_classes]), name="weight")
  b = tf.Variable(tf.random_normal([nb_classes]), name="bias")

  # softmax = 
  logits = tf.matmul(X, W) + b
  hypothesis = tf.nn.softmax(logits)

  # Cross entropy cost / loss
  # With softmax_cross_entropy_*with_logits
  cost_i = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y_one_hot)
  cost = tf.reduce_mean(cost_i)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

  prediction = tf.argmax(hypothesis, 1)
  correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  feed = {X: x_data, Y: y_data}
  # launch graph
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
      sess.run(optimizer, feed_dict=feed)
      if step % 100 == 0:
        loss, acc = sess.run([cost, accuracy], feed_dict=feed)
        print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    # y_data : (N, 1) = flatten >= (N, ) matches pred.shape
    for p, y in zip(pred, y_data.flatten()):
      print("[{}] prediction: {} True Y: {}".format(p == int(y), p, int(y)))

if __name__ == "__main__":
  # prac01()
  prac02()