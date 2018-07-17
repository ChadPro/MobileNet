import tensorflow as tf
from nets import nets_factory

mobile_net = nets_factory.get_network('mobile_net_300')

img = tf.ones([1,300,300,3])

y, _, __ = mobile_net(img)

with tf.Session() as sess:
    variable_init = tf.global_variables_initializer()
    sess.run(variable_init)

    result = sess.run(y)
    print result.shape