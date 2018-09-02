import tensorflow as tf
import numpy as np
training_file = ["C:\\Users\\deepak03\\Downloads\\train_mnist.csv"]
record_defaults = [tf.double]*785


a = []
for i in range(785):
    a.append(i)
dataset = tf.contrib.data.CsvDataset(training_file,record_defaults,header = True,select_cols=a)
dataset = dataset.batch(20)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
sess = tf.Session()
for i in range(10):
    value = sess.run(tf.transpose(next_element))
    print(value[:,1:786])
    print(np.shape(value[:,1:788]))
