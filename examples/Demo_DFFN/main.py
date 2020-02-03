"""
"""

import os
import random

import argparse
import numpy as np
import tensorflow as tf

from data import load_data, multiversion_matfile_get_field, get_train_val_splits, nclass_dict, dset_dims, dset_filenames_dict, dset_fieldnames_dict, pca_embedding
from models.train_indian_pines import DFFN_indian_pines


import pdb

PAD_TYPE = 'symmetric'
learning_rate = 0.1

network_dict = {
    'KSC': DFFN_indian_pines,
    'IP': DFFN_indian_pines
}

def make_one_hot(indices, n_classes):
    n_examples = indices.shape[0]
    indices = np.random.randint(0,n_classes,(n_examples,))
    one_hot = np.zeros((indices.size, indices.max()+1))
    one_hot[np.arange(indices.size),indices] = 1
    return one_hot

def train(args):
    bs = args.batch_size
    n_classes = nclass_dict[args.dataset]
    trainimgname, trainlabelname = dset_filenames_dict[args.dataset]
    trainimgfield, trainlabelfield = dset_fieldnames_dict[args.dataset]
    
    
    # tf ops
    images = tf.placeholder(tf.float32, [bs, 25, 25, 3])
    labels = tf.placeholder(tf.float32, [bs, n_classes])
    net = DFFN_indian_pines({'data': images})
    logits = net.layers['InnerProduct1']
    
    pred_classes = tf.argmax(logits, axis=1)
    
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels), 0)
            
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate)
    train_op = optimizer.minimize(loss_op) # ,global_step=tf.train.get_global_step()

    # acc_op = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=pred_classes)
    # end of tf ops
        
    train_mask = multiversion_matfile_get_field(args.mask_root, 'train_mask')
    val_mask = multiversion_matfile_get_field(args.mask_root, 'test_mask')
    
    data, labels_ = load_data(trainimgname, trainimgfield, trainlabelname, trainlabelfield)
    
    data = pca_embedding(data, n_components=args.network_spectral_size)
            
    s = args.network_spatial_size - 1
    trainX, trainY, valX, valY = get_train_val_splits(data, labels_, train_mask, val_mask, (s,s,0), n_eval=args.n_eval)
    
    best_loss = float("inf")
    best_acc = 0
    train_set_size = trainX.shape[0]
    val_set_size = valX.shape[0]
    steps_per_epoch = train_set_size // bs
    max_steps = args.num_epochs * steps_per_epoch
    
    trainY = make_one_hot(trainY, n_classes)
    valY = make_one_hot(valY, n_classes)
    
    train_idxs = np.arange(trainX.shape[0])
    itr_per_epoch = len(train_idxs) // bs

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        eval_loss = 0
        eval_acc = 0
        for j in range(valX.shape[0] // bs):
            labs = valY[(j*bs):((j+1)*bs)]
            feed = {images: valX[(j*bs):((j+1)*bs)], labels: labs }
            np_loss, np_pred = sess.run([loss_op, pred_classes], feed_dict=feed)
            eval_loss += np_loss / itr_per_epoch
            acc = 100* (np_pred == labs.argmax(axis=1)).sum() / float(bs)
            eval_acc += acc / (valX.shape[0] // bs)
        print("{:06d}: Train Loss: {:.4f}. Eval Loss: {:.4f}. Val Acc: {:.4f} (Best: {:.4f})".format(0, 0, eval_loss, eval_acc, 0))
        
        net.load('/scratch0/ilya/locDownloads/indian_pines2.npy', sess)
        eval_loss = 0
        eval_acc = 0
        for j in range(valX.shape[0] // bs):
            labs = valY[(j*bs):((j+1)*bs)]
            feed = {images: valX[(j*bs):((j+1)*bs)], labels: labs }
            np_loss, np_pred = sess.run([loss_op, pred_classes], feed_dict=feed)
            eval_loss += np_loss / itr_per_epoch
            acc = 100* (np_pred == labs.argmax(axis=1)).sum() / float(bs)
            eval_acc += acc / (valX.shape[0] // bs)
            
        print("{:06d}: Train Loss: {:.4f}. Eval Loss: {:.4f}. Val Acc: {:.4f} (Best: {:.4f})".format(0, 0, eval_loss, eval_acc, 0))
        
        # for i in range(args.num_epochs):
        #     random.shuffle(train_idxs)
        #     train_loss = 0
        #     train_acc = 0
        #     eval_loss = 0
        #     eval_acc = 0
            
        #     for j in range(itr_per_epoch):
        #         labs = trainY[ train_idxs[(j*bs):((j+1)*bs)] ]
        #         feed = {images: trainX[ train_idxs[(j*bs):((j+1)*bs)] ], labels: labs }
        #         np_loss, np_pred, _ = sess.run([loss_op, pred_classes, train_op], feed_dict=feed)
        #         train_loss += np_loss / itr_per_epoch
        #         acc = 100* (np_pred == labs.argmax(axis=1)).sum() / float(bs)
        #         train_acc += acc / itr_per_epoch
                
        #     # eval
        #     if i % args.eval_period == 0:
        #         for j in range(valX.shape[0] // bs):
        #             labs = valY[(j*bs):((j+1)*bs)]
        #             feed = {images: valX[(j*bs):((j+1)*bs)], labels: labs }
        #             np_loss, np_pred = sess.run([loss_op, pred_classes], feed_dict=feed)
        #             eval_loss += np_loss / itr_per_epoch
        #             acc = 100* (np_pred == labs.argmax(axis=1)).sum() / float(bs)
        #             eval_acc += acc / (valX.shape[0] // bs)
                
            
        #         best_loss = min(best_loss, train_loss)
        #         best_acc = max(best_acc, eval_acc)
        #         print("{:06d}: Train Loss: {:.4f}. Eval Loss: {:.4f}. Val Acc: {:.4f} (Best: {:.4f})".format(i, train_loss, eval_loss, eval_acc, best_acc))
        
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--dataset', required=True,
                      help='Name of dataset to run on')
    parser.add_argument('--model_root', required=True,
                      help='Full path of where to output the results of training.')
    parser.add_argument('--mask_root', default=None,
                      help='Required unless supplying --predict. Full path to mask to use to generate train/val set.')
    # Optionals
    parser.add_argument(
        '--batch_size', type=int, default=100,
        help='Batch Size')
    parser.add_argument(
        '--num_epochs', type=int, default=10000,
        help='Number of epochs to run training for.')
    parser.add_argument(
        '--eval_period', type=int, default=50,
        help='Eval after every N epochs.')
    parser.add_argument(
        '--predict', action='store_true', default=False,
        help='If true predict on the whole HSI image (default: %(default)s)')
    parser.add_argument(
        '--n_eval', type=int, default=int(1e12),
        help='Restrict size of the eval set during training. Useful when input is preprocessed for ST')
    parser.add_argument(
        '--network_spatial_size', type=int, default=25,
        help='Size of image patch to pass to DFFN. IP=25, PaviaU=23, Salinas=27.')
    parser.add_argument(
        '--network_spectral_size', type=int, default=3,
        help='Num channels to pass to DFFN (projected down to with PCA). IP=3, PaviaU=5, Salinas=10.')
    args = parser.parse_args()

    if not os.path.isdir(args.model_root):
        os.makedirs(args.model_root)

    train(args)


if __name__ == '__main__':
    main()