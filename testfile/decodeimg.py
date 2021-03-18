import tensorflow as tf

img = tf.io.read_file("./data/img/23095658544_7226386954_n.jpg")
image = tf.image.decode_jpeg(img)
print(image)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    # image = tf.image.resize(image, [im_height, im_width])