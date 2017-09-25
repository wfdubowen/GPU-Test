import tensorflow as tf

a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a' )
b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b' )
c = a + b

sess = tf.Session(config=tf.ConfigProto(log_device_placement = True))
print(sess.run(c))