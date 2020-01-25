"""
"""

import os

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

def train(args):
    bs = args.batch_size
    
    def network(x_dict):
        net = network_dict[args.dataset](x_dict)
        return net.layers['InnerProduct1']
        
    n_classes = nclass_dict[args.dataset]
    trainimgname, trainlabelname = dset_filenames_dict[args.dataset]
    trainimgfield, trainlabelfield = dset_fieldnames_dict[args.dataset]
    
    train_mask = multiversion_matfile_get_field(args.mask_root, 'train_mask')
    val_mask = multiversion_matfile_get_field(args.mask_root, 'test_mask')
    
    data, labels = load_data(trainimgname, trainimgfield, trainlabelname, trainlabelfield)
    
    data = pca_embedding(data, n_components=args.network_spectral_size)
            
    s = args.network_spatial_size - 1
    trainX, trainY, valX, valY = get_train_val_splits(data, labels, train_mask, val_mask, (s,s,0), n_eval=args.n_eval)
    
    best_loss = float("inf")
    best_acc = 0
    train_set_size = trainX.shape[0]
    val_set_size = valX.shape[0]
    steps_per_epoch = train_set_size // bs
    max_steps = args.num_epochs * steps_per_epoch
    
    def model_fn(features, labels, mode):
        logits_train = network(features)
        logits_val = network(features)
    
        # Predictions
        pred_classes = tf.argmax(logits_val, axis=1)
        pred_probas = tf.nn.softmax(logits_val)
    
        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)
    
            # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
        
        # could also try tensorflow addons
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        train_op = optimizer.minimize(loss_op,
                                      global_step=tf.train.get_global_step())
    
        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    
        # tf.summary.scalar('min', loss_op)
    
        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op})
    
        return estim_specs
    
    model = tf.estimator.Estimator(model_fn, model_dir=args.model_root)
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'data': trainX[:,:,:,:]}, y=trainY[:],
        batch_size=bs, num_epochs=args.eval_period, shuffle=True)
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'data': valX[:args.n_eval,:,:,:]}, y=valY[:args.n_eval],
        batch_size=bs, shuffle=False)
    
    for i in range(args.num_epochs // args.eval_period):
        model.train(train_input_fn)

        e = model.evaluate(eval_input_fn)
        
        best_loss = min(best_loss, e['loss'])
        best_acc = max(best_acc, e['accuracy'])
        print("{:06d}: Validation Accuracy: {:.4f} (Best: {:.4f})".format(i*args.eval_period, e['accuracy'], best_acc))
        
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