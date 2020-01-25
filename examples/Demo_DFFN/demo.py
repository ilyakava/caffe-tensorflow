# Import the converted model's class
import numpy as np
import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from models.train_indian_pines import DFFN_indian_pines as DFFN

import pdb

bs = 100
n_classes = 16

images = tf.placeholder(tf.float32, [bs, 25, 25, 3])
labels = tf.placeholder(tf.float32, [bs, n_classes])
net = DFFN({'data': images})

logits = net.layers['InnerProduct1']
pred = tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels), 0)
opt = tf.train.RMSPropOptimizer(0.001)
train_op = opt.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # TODO: initialize properly?
    
    indices = np.random.randint(0,n_classes,(bs,))
    one_hot = np.zeros((indices.size, indices.max()+1))
    one_hot[np.arange(indices.size),indices] = 1
    
    
    feed = {images: np.random.random((bs, 25, 25, 3)), labels: one_hot }
    np_loss, np_pred, _ = sess.run([loss, pred, train_op], feed_dict=feed)
    print('Fed some random data in with loss: ', np_loss)
