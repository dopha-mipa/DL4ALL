import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def prac01():
  t = np.array([[1., 2., 3.],
                [4., 5., 6.],
                [7., 8., 9.],
                [10., 11., 12.],
                [13., 14., 15.]])
  pp.pprint(t)
  print(t.ndim) # rank
  print(t.shape) # shape

  t = tf.constant([1, 2, 3, 4])
  tf.shape(t).eval() # array([4], dtype=int32)

  t = tf.constant([[1, 2], [3, 4]])
  tf.shape(t).eval() # array([2, 2], dtype=int32)

  t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [7, 8, 9, 10]],
                    [[11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22]]]])
  tf.shape(t).eval() # array([1, 2, 3, 4], dtype=int32)

  # shape는 모양, rank는 열의 수, axis는 몇 겹인지..? 0부터 세는 듯 하다..
  # 제일 안쪽의 axis는 그냥 파이썬 처럼 -1이라고 하기도 한다. 아!

  matrix1 = tf.constant([[1., 2.], [3., 4.]]) # 2x2
  matrix2 = tf.constant([[1.], [3.]]) # 2*1
  tf.matmul(matrix1, matrix2).eval() # 그냥 곱하면 안돼요!! 
  # matrix1 * matrix2 는 결과가 안나오는게 아니라 산으로 감..

  # Broadcasting : 행렬모양이 같지 않아도 연산해주는 것!

  x = [[1., 2.], [3., 4.]]
  tf.reduce_mean(x).eval() # 2.5
  tf.reduce_mean(x, axis=0).eval() # [2., 3.]
  tf.reduce_mean(x, axis=1).eval() # [1.5, 3.5]
  tf.reduce_mean(x, axis=-1).eval() # 가장 안쪽 axis!![1.5, 3.5]

  x = [[0., 1., 2.],
       [2., 1., 0.]]
  tf.argmax(x, axis=0).eval() # index: [1, 0, 0]
  tf.argmax(x, axis=1).eval() # index: [2, 0]
  tf.argmax(x, axis=-1).eval() # index: [2, 0]

  # ★★ Reshape ★★
  t = np.array([[[0, 1, 2],
                 [3, 4, 5]],
                [[6, 7, 8],
                 [9, 10, 11]]])
  t.shape # (2, 2, 3)
  tf.reshape(t, shape=[-1, 3]).eval()
  # 랭크 두 개에.. -1은 네가 알아서 해?;3은 제일 안쪽의 그 값. 
  # [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]

  tf.reshape(t, shape=[-1, 1, 3]).eval()
  # [[[0, 1, 2]], [[3, 4, 5]], [[6, 7, 8]], [[9, 10, 11]]]

  tf.squeeze([[0], [1], [2]]).eval()
  # [0, 1, 2]
  tf.expand_dims([0, 1, 2], 1).eval()
  # [[0], [1], [2]]

  # one-hot!
  t = tf.one_hot([[0], [1], [2], [0]], depth=3).eval()
  # [[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]]
  
  tf.reshape(t, shape=[-1, 3]).eval()
  # [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0],]

  tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval()
  # [1, 2, 3, 4]
  tf.cast([True, False, 1 == 1, 0 == 1], tf.int32)
  # [1, 0, 1, 0]

  # Stack
  x = [1, 4]
  y = [2, 5]
  z = [3, 6]
  # Pack along first dim
  tf.stack([x, y, z]).eval()
  # [[1, 4], [2, 5], [3, 6]]
  tf.stack([x, y, z], axis=1).eval()
  # [[1, 2, 3], [4, 5, 6]]

  x = [[0, 1, 2], [2, 1, 0]]
  tf.ones_like(x).eval()
  # [[1, 1, 1], [1, 1, 1]]
  tf.zeros_like(x).eval()
  # [[0, 0, 0], [0, 0, 0]]

  for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)

if __name__ == "__main__":
    main()
