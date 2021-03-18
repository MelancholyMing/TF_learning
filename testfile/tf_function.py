import tensorflow as tf
import numpy as np


@tf.function
def f(x):
    print("The function is running in Python")
    tf.print(x)

a = tf.constant(1, dtype=tf.int32)
f(a)
b = tf.constant(2, dtype=tf.int32)
f(b)
b_ = np.array(2, dtype=np.int32)
f(b_)
c = tf.constant(0.1, dtype=tf.float32)
f(c)
d = tf.constant(0.2, dtype=tf.float32)
f(d)


# def outer(func):
#     def inner():
#         print("我是内层函数！")
#
#     return inner()
#
#
# def foo():
#     print("我是原始函数！")
#
# # outer(foo)
# outer(foo())