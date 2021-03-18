from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
# print(gpus, cpus)
#

"""
二、日志设备放置（Logging device placement）
    为了查出我们的操作和张量被配置到哪个设备上，我们可以
    将‘tf.debugging.set_log_device_placement (True)’作为你程序的第一个表达。
"""
# tf.debugging.set_log_device_placement(True)
#
# # Create some tensors
# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)
#
# print(c)
#
"""
三、手动设备配置（Manual device placement）
    如果希望在自己选择的设备上运行特定的操作，而不是自动为我们选择的操作，
    我们可以使用 tf.device 来创建设备上下文，该上下文中的所有操作都将在相同的指定设备上运行。
"""
# with tf.device('/CPU:0'):
#     a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#     b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#
# c = tf.matmul(a, b)
# print(c)

"""
四、限制 GPU 内存增长（Limiting GPU memory growth）
    默认情况下，TensorFlow 会将所有 GPU (取决于 CUDA_VISIBLE_DEVICES) 的几乎所有 GPU 内存映射到进程。
    这样做是为了通过减少内存碎片更有效地使用设备上相对宝贵的 GPU 内存资源。为了将 TensorFlow 限制在一组特定的 gpu 上，
    我们使用 tf.config.experimental.set_visible_devices 方法。
"""
# if gpus:
#     # Restrict TensorFlow to only use the first GPU
#     try:
#         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#     except RuntimeError as e:
#         # Visible devices must be set before GPUs have been initialized
#         print(e)

"""
    在某些情况下，希望进程只分配可用内存的一个子集，或者只根据进程的需要增加内存使用量。TensorFlow 提供了两种方法来控制它。
   第一个选项是通过调用 tf.config.experimental.set_memory_growth 来打开内存增长，它试图只分配运行时所需的 GPU 内存：
   它开始分配非常少的内存，随着程序运行和更多的 GPU 内存需要，我们扩展分配给 TensorFlow 进程的 GPU 内存区域。注意，我们不
   释放内存，因为它会导致内存碎片。要为特定的 GPU 打开内存增长，请在分配任何张量或执行任何操作之前使用以下代码。
   激活这条选项的另外一种方式是设置环境变量‘TF_FORCE_GPU_ALLOW_GROTH’为‘True’。
"""

# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)

"""
    第二个方法是用‘tf.config.experimental.set_virtual_device_configuration’配置一个虚拟 GPU 设备并
    严格限制分配给 GPU 的内存。
"""
# if gpus:
#     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)


"""
五、在多 GPU 系统上使用单个 GPU
    如果我们的系统里有不止一个 GPU，则默认情况下，ID 最小的 GPU 将被选用。如果想在不同的 GPU 上运行，我们
    需要显式地指定优先项。
"""
# tf.debugging.set_log_device_placement(True)
#
# try:
#     # Specify an invalid GPU device
#     with tf.device('/device:GPU:2'):
#         a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#         b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#         c = tf.matmul(a, b)
#         print(c)
# except RuntimeError as e:
#     print(e)

"""
    如果希望 TensorFlow 自动选择一个现有且受支持的设备来运行操作，以避免指定的设备不存在，那么可以调
    用 tf.config.set_soft_device_placement (True)。
"""

# tf.config.set_soft_device_placement(True)
# tf.debugging.set_log_device_placement(True)
#
# # Creates some tensors
# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)
#
# print(c)

"""
六、使用多 GPU
    为多个 gpu 开发将允许模型使用额外的资源进行扩展。如果在一个系统上用一个 GPU 开发，我们可以用虚拟设
    备模拟多个 GPU。这使得测试多 gpu 设置变得容易，而不需要额外的资源。
    当我们有多个本地 GPU 用来运行时，我们可以用‘tf.distribute.Strategy’或手动配置这些 GPU。
"""

# if gpus:
#   # Create 2 virtual GPUs with 1GB memory each
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
#          tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)

"""
6.1 使用‘tf.distribute.Strategy’
    使用多 GPU 的最佳实践是用‘tf.distribute.Strategy’。下面是个简单的例子：
"""
# tf.debugging.set_log_device_placement(True)
#
# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
#     inputs = tf.keras.layers.Input(shape=(1,))
#     predictions = tf.keras.layers.Dense(1)(inputs)
#     model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
#     model.compile(loss='mse',
#                   optimizer=tf.keras.optimizers.SGD(learning_rate=0.2))


"""
6.2 手动配置
    tf.distribute.Strategy 通过在幕后跨设备复制计算，这样工作。我们可以通过在每个 GPU 上构建模
    型来手动实现复制。    
"""
tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_logical_devices('GPU')

if gpus:
    # Replicate your computation on multiple GPUs
    c = []
    for gpu in gpus:
        with tf.device(gpu.name):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c.append(tf.matmul(a, b))

    with tf.device('/CPU:0'):
        matmul_sum = tf.add_n(c)

    print(matmul_sum)
