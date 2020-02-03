import sys
sys.path.append('../..')

from kaffe.tensorflow import Network

import tensorflow as tf

class DFFN_indian_pines(Network):
    def setup(self):
        # with tf.variable_scope('Hyper3DNet', reuse=tf.AUTO_REUSE):
        (self.feed('data')
             .conv(3, 3, 16, 1, 1, relu=False, name='Convolution1')
             .batch_normalization(relu=True, name='BatchNorm1')
             .conv(3, 3, 16, 1, 1, relu=False, name='Convolution2')
             .batch_normalization(relu=True, name='BatchNorm2')
             .conv(3, 3, 16, 1, 1, relu=False, name='Convolution3')
             .batch_normalization(name='BatchNorm3'))

        (self.feed('BatchNorm3', 
                   'BatchNorm1')
             .add(name='Eltwise1')
             .relu(name='ReLU3')
             .conv(3, 3, 16, 1, 1, relu=False, name='Convolution4')
             .batch_normalization(relu=True, name='BatchNorm4')
             .conv(3, 3, 16, 1, 1, relu=False, name='Convolution5')
             .batch_normalization(name='BatchNorm5'))

        (self.feed('ReLU3', 
                   'BatchNorm5')
             .add(name='Eltwise2')
             .relu(name='ReLU5')
             .conv(3, 3, 16, 1, 1, relu=False, name='Convolution6')
             .batch_normalization(relu=True, name='BatchNorm6')
             .conv(3, 3, 16, 1, 1, relu=False, name='Convolution7')
             .batch_normalization(name='BatchNorm7'))

        (self.feed('ReLU5', 
                   'BatchNorm7')
             .add(name='Eltwise3')
             .relu(name='ReLU7')
             .conv(3, 3, 16, 1, 1, relu=False, name='Convolution8')
             .batch_normalization(relu=True, name='BatchNorm8')
             .conv(3, 3, 16, 1, 1, relu=False, name='Convolution9')
             .batch_normalization(name='BatchNorm9'))

        (self.feed('ReLU7', 
                   'BatchNorm9')
             .add(name='Eltwise4')
             .relu(name='ReLU9')
             .conv(1, 1, 32, 2, 2, relu=False, name='Convolution10')
             .batch_normalization(name='BatchNorm10'))

        (self.feed('ReLU9')
             .conv(3, 3, 32, 2, 2, relu=False, name='Convolution11')
             .batch_normalization(relu=True, name='BatchNorm11')
             .conv(3, 3, 32, 1, 1, relu=False, name='Convolution12')
             .batch_normalization(name='BatchNorm12'))

        (self.feed('BatchNorm10', 
                   'BatchNorm12')
             .add(name='Eltwise5')
             .relu(name='ReLU11')
             .conv(3, 3, 32, 1, 1, relu=False, name='Convolution13')
             .batch_normalization(relu=True, name='BatchNorm13')
             .conv(3, 3, 32, 1, 1, relu=False, name='Convolution14')
             .batch_normalization(name='BatchNorm14'))

        (self.feed('ReLU11', 
                   'BatchNorm14')
             .add(name='Eltwise6')
             .relu(name='ReLU13')
             .conv(3, 3, 32, 1, 1, relu=False, name='Convolution15')
             .batch_normalization(relu=True, name='BatchNorm15')
             .conv(3, 3, 32, 1, 1, relu=False, name='Convolution16')
             .batch_normalization(name='BatchNorm16'))

        (self.feed('ReLU13', 
                   'BatchNorm16')
             .add(name='Eltwise7')
             .relu(name='ReLU15')
             .conv(3, 3, 32, 1, 1, relu=False, name='Convolution17')
             .batch_normalization(relu=True, name='BatchNorm17')
             .conv(3, 3, 32, 1, 1, relu=False, name='Convolution18')
             .batch_normalization(name='BatchNorm18'))

        (self.feed('ReLU15', 
                   'BatchNorm18')
             .add(name='Eltwise8')
             .relu(name='ReLU17')
             .conv(1, 1, 64, 2, 2, relu=False, name='Convolution19')
             .batch_normalization(name='BatchNorm19'))

        (self.feed('ReLU17')
             .conv(3, 3, 64, 2, 2, relu=False, name='Convolution20')
             .batch_normalization(relu=True, name='BatchNorm20')
             .conv(3, 3, 64, 1, 1, relu=False, name='Convolution21')
             .batch_normalization(name='BatchNorm21'))

        (self.feed('BatchNorm19', 
                   'BatchNorm21')
             .add(name='Eltwise9')
             .relu(name='ReLU19')
             .conv(3, 3, 64, 1, 1, relu=False, name='Convolution22')
             .batch_normalization(relu=True, name='BatchNorm22')
             .conv(3, 3, 64, 1, 1, relu=False, name='Convolution23')
             .batch_normalization(name='BatchNorm23'))

        (self.feed('ReLU19', 
                   'BatchNorm23')
             .add(name='Eltwise10')
             .relu(name='ReLU21')
             .conv(3, 3, 64, 1, 1, relu=False, name='Convolution24')
             .batch_normalization(relu=True, name='BatchNorm24')
             .conv(3, 3, 64, 1, 1, relu=False, name='Convolution25')
             .batch_normalization(name='BatchNorm25'))

        (self.feed('ReLU21', 
                   'BatchNorm25')
             .add(name='Eltwise11')
             .relu(name='ReLU23')
             .conv(3, 3, 64, 1, 1, relu=False, name='Convolution26')
             .batch_normalization(relu=True, name='BatchNorm26')
             .conv(3, 3, 64, 1, 1, relu=False, name='Convolution27')
             .batch_normalization(name='BatchNorm27'))

        (self.feed('ReLU23', 
                   'BatchNorm27')
             .add(name='Eltwise12'))

        (self.feed('ReLU9')
             .conv(3, 3, 64, 4, 4, relu=False, name='Convolution_eltwise4')
             .batch_normalization(name='BatchNorm_Convolution_eltwise4'))

        (self.feed('ReLU17')
             .conv(3, 3, 64, 2, 2, relu=False, name='Convolution_eltwise8')
             .batch_normalization(name='BatchNorm_Convolution_eltwise8'))

        (self.feed('BatchNorm_Convolution_eltwise4', 
                   'BatchNorm_Convolution_eltwise8')
             .add(name='fuse1'))

        (self.feed('fuse1', 
                   'Eltwise12')
             .add(name='fuse2')
             .relu(name='ReLU25')
             .avg_pool(7, 7, 1, 1, padding='VALID', name='Pooling1')
             .fc(16, relu=False, name='InnerProduct1'))