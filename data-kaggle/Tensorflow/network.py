import tensorflow as tf

training_file = ["C:\\Users\\deepak03\\Downloads\\train_mnist.csv"]
record_defaults = [tf.double]*785

a = []
for i in range(785):
    a.append(i)
dataset = tf.contrib.data.CsvDataset(training_file,record_defaults,header = True,select_cols=a)
dataset = dataset.batch(10)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
value = tf.transpose(next_element)
input_placeholder = value[:,1:785]
y_=tf.cast(value[:,:1],dtype=tf.int32)
'''
y_=value[:,:1]
print(y_.get_shape())

'''


l1_w1 = tf.get_variable("l1_w1",[784,784],dtype=tf.float64)
l1_mat = tf.matmul(input_placeholder,l1_w1)
l2_w2 = tf.get_variable("l1_w2",[784,10],dtype=tf.float64)
label_out = tf.nn.softmax(tf.matmul(l1_mat,l2_w2))
print(label_out)


'''
loss = tf.losses.mean_squared_error( label_out, y_, weights=1.0, scope=None, loss_collection=tf.GraphKeys.LOSSES)
train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
'''
init_all_op = tf.global_variables_initializer()

sess = tf.Session()
itr = sess.run(iterator.initializer)
sess.run(init_all_op)

for i in range(1):
    print(sess.run(label_out))
    print(sess.run(tf.one_hot(y_,10)))
    print(sess.run(y_))
    print(y_.get_shape())
    print(tf.one_hot(y_,10).get_shape())
    print(label_out.get_shape())


