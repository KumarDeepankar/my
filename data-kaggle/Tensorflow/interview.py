import tensorflow as tf
# Build your graph.
x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(([2.0, 2.0],[3.0,3.0]))
y = tf.matmul(x, w)
l = [[1000,10],[100,100]]

loss = tf.losses.mean_squared_error( l, y, weights=1.0, scope=None, loss_collection=tf.GraphKeys.LOSSES )
train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
init_all_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_all_op)

for i in range(15):
    sess.run(train_op)
print(sess.run(w))
