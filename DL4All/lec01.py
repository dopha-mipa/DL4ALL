'''
# 2017. 04. 17 ~ 
# 모두를 위한 딥러닝
https://docs.google.com/presentation/d/137IlT2N3AYcclqxNuc8j9RDrIeHiYkSZ5JPg_vg9Jqk/edit#slide=id.g1d115b0ec5_0_0
'''
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def prac01():
  # Build graph - 그래프를 정의한다
  node1 = tf.constant(3.0, tf.float32)
  node2 = tf.constant(4.0) # also tf.float32 implicitly
  node3 = tf.add(node1, node2)

  print("node1 : ", node1, "\nnode2 : ", node2)
  print("node 3 : ", node3)

  # Feed data and run graph
  sess = tf.Session()

  # Update variables in the graph
  print(sess.run([node1, node2]))
  print(sess.run(node3))

  '''
  slide note : 
  “placeholder는 레이지 기법으로 값을 미리 정하지 않고, 
  프로그래밍이 돌아가는 시점에 정해지는 값입니다. 
  변수의 형태(type)만 정해 놓고, 
  변수의 값은 정해지지 않았지만 텐서플로우가 실행되는 동안에 정해집니다. 
  값이 미리 정해지지 않고, 나중에 정해지는 점은 
  굉장히 유연한 프로그래밍을 할 수 있게 도와줍니다.”
  '''
  a = tf.placeholder(tf.float32)
  b = tf.placeholder(tf.float32)
  adder_node = a + b

  print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
  print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))

  # rank와 Shape을 잘 구분하자!
  # rank : scalar, vector, matrix, 
  # 3-Tensor([[[2], [3], [4]], [[5], [6], [7]], [[...]]], n-Tensor..
  # shape : [A1, A2, A3, ..... , An] = n-Dimension
  # shape은 tensor를 설계할때 중요해 ~ 
  # type : float32/64, int8/16/32/64 - float32가 주고 사용됨

if __name__ == "__main__":
  prac01()