import tensorflow as tf
# tensorflow.org/get_started/mnist/beginners
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.set_random_seed(777)  # for reproducibility

# ☆ MNIST Dataset!
# 미안.. 미안해...
# 정확도를 높인다는건 어렵네
# 텐서보드를 쓰면 배로 오래 걸린다..
def prac01():
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

  # 0 ~ 9 : 10 classes
  nb_classes = 10
  # 28 * 28 shape
  X = tf.placeholder(tf.float32, [None, 784])
  Y = tf.placeholder(tf.float32, [None, nb_classes]) # one_hot

  # Accuracy : 0.9001
  # 하고 epochs 12으로 늘렸더니 0.05쯤 좋아졌다. 
  # learningrate=0.1 이었다. 좀 크네

  training_epochs = 10
  batch_size = 50 # 한번에 학습하는 사이즈를 줄인 게 효과가 좋은 것 같다..!
  # 일단 제일 첫 코스트가 2점대에서 1.3점 대로 낮아졌어 무려
  # 어제 앨리스에서 좀 나이브하게 하는걸 보기도 했고.. 
  # wide nn 검색해서 배치 50짜리를 보기도 했고

  with tf.name_scope("layer1") as scope:
    # 느낌적으로는.. 784개 특성에 반응하는 뉴런에 최소 1 + a개 있게 하고
    W1 = tf.Variable(tf.random_normal([784, 1000]))
    b1 = tf.Variable(tf.random_normal([1000]))
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    w1_hist = tf.summary.histogram("weights1", W1)
    b1_hist = tf.summary.histogram("bias1", b1)
    layer1_hist = tf.summary.histogram("layer1", layer1)

  with tf.name_scope("layer2") as scope:
    # 그 특성 중에 쓸만한 애들을 거른? 다음
    W2 = tf.Variable(tf.random_normal([1000, 100]))
    b2 = tf.Variable(tf.random_normal([100]))
    layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    w2_hist = tf.summary.histogram("weights2", W2)
    b2_hist = tf.summary.histogram("bias2", b2)
    layer2_hist = tf.summary.histogram("layer2", layer2)

  # 최종적으로 클래스 수에 맞도록 해봤다...
  with tf.name_scope("layer3") as scope:
    W3 = tf.Variable(tf.random_normal([100, nb_classes]))
    b3 = tf.Variable(tf.random_normal([nb_classes]))
    w3_hist = tf.summary.histogram("weights3", W3)
    b3_hist = tf.summary.histogram("bias3", b3)

    hypothesis = tf.nn.softmax(tf.matmul(layer2, W3) + b3)
    hypo_hist = tf.summary.histogram("hypothesis", hypothesis)

  cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

  cost_summ = tf.summary.scalar("cost", cost)

  # Test model
  is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
  # Calculate accuracy
  accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

  # batch로 잘라서 학습시키자! 
  # epoch : 전체 데이터 셋을 한번 다 학습시키는 것을 epoch라고 한다.

  summary = tf.summary.merge_all()

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Create summary writer
    writer = tf.summary.FileWriter("./")
    writer.add_graph(sess.graph)

    # Training cycle
    for epoch in range(training_epochs):
      avg_cost = 0
      total_batch = int(mnist.train.num_examples / batch_size)
      global_step = 0

      for i in range(total_batch):
        # Q : data를 100개씩 읽어들이자... 이건 가중치 계속 바꿀테니까 온라인인가..?
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, s, _ = sess.run([cost, summary, optimizer], 
            feed_dict={X: batch_xs, Y: batch_ys})
        writer.add_summary(s, global_step=global_step)
        global_step += 1
        avg_cost += c / total_batch

      print("Epoch:", "%04d" % (epoch + 1),
            "cost =", "{:.9f}".format(avg_cost))
    
    writer = tf.summary.FileWriter("./logs/xor_logs")
    # 하고 cmd에서 tensorboard -logdir=./logs/xor_logs
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
  prac01()