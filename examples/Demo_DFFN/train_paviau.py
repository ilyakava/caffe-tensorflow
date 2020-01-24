from kaffe.tensorflow import Network

class DFFN_paviau(Network):
    def setup(self):
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
             .conv(3, 3, 16, 1, 1, relu=False, name='Convolution10')
             .batch_normalization(relu=True, name='BatchNorm10')
             .conv(3, 3, 16, 1, 1, relu=False, name='Convolution11')
             .batch_normalization(name='BatchNorm11'))

        (self.feed('ReLU9', 
                   'BatchNorm11')
             .add(name='Eltwise5')
             .relu(name='ReLU11')
             .conv(1, 1, 32, 2, 2, relu=False, name='Convolution18')
             .batch_normalization(name='BatchNorm18'))

        (self.feed('ReLU11')
             .conv(3, 3, 32, 2, 2, relu=False, name='Convolution19')
             .batch_normalization(relu=True, name='BatchNorm19')
             .conv(3, 3, 32, 1, 1, relu=False, name='Convolution20')
             .batch_normalization(name='BatchNorm20'))

        (self.feed('BatchNorm18', 
                   'BatchNorm20')
             .add(name='Eltwise9')
             .relu(name='ReLU19')
             .conv(3, 3, 32, 1, 1, relu=False, name='Convolution21')
             .batch_normalization(relu=True, name='BatchNorm21')
             .conv(3, 3, 32, 1, 1, relu=False, name='Convolution22')
             .batch_normalization(name='BatchNorm22'))

        (self.feed('ReLU19', 
                   'BatchNorm22')
             .add(name='Eltwise10')
             .relu(name='ReLU21')
             .conv(3, 3, 32, 1, 1, relu=False, name='Convolution23')
             .batch_normalization(relu=True, name='BatchNorm23')
             .conv(3, 3, 32, 1, 1, relu=False, name='Convolution24')
             .batch_normalization(name='BatchNorm24'))

        (self.feed('ReLU21', 
                   'BatchNorm24')
             .add(name='Eltwise11')
             .relu(name='ReLU23')
             .conv(3, 3, 32, 1, 1, relu=False, name='Convolution25')
             .batch_normalization(relu=True, name='BatchNorm25')
             .conv(3, 3, 32, 1, 1, relu=False, name='Convolution26')
             .batch_normalization(name='BatchNorm26'))

        (self.feed('ReLU23', 
                   'BatchNorm26')
             .add(name='Eltwise12')
             .relu(name='ReLU25')
             .conv(3, 3, 32, 1, 1, relu=False, name='Convolution27')
             .batch_normalization(relu=True, name='BatchNorm27')
             .conv(3, 3, 32, 1, 1, relu=False, name='Convolution28')
             .batch_normalization(name='BatchNorm28'))

        (self.feed('ReLU25', 
                   'BatchNorm28')
             .add(name='Eltwise13')
             .relu(name='ReLU27')
             .conv(1, 1, 64, 2, 2, relu=False, name='Convolution35')
             .batch_normalization(name='BatchNorm35'))

        (self.feed('ReLU27')
             .conv(3, 3, 64, 2, 2, relu=False, name='Convolution36')
             .batch_normalization(relu=True, name='BatchNorm36')
             .conv(3, 3, 64, 1, 1, relu=False, name='Convolution37')
             .batch_normalization(name='BatchNorm37'))

        (self.feed('BatchNorm35', 
                   'BatchNorm37')
             .add(name='Eltwise17')
             .relu(name='ReLU35')
             .conv(3, 3, 64, 1, 1, relu=False, name='Convolution38')
             .batch_normalization(relu=True, name='BatchNorm38')
             .conv(3, 3, 64, 1, 1, relu=False, name='Convolution39')
             .batch_normalization(name='BatchNorm39'))

        (self.feed('ReLU35', 
                   'BatchNorm39')
             .add(name='Eltwise18')
             .relu(name='ReLU37')
             .conv(3, 3, 64, 1, 1, relu=False, name='Convolution40')
             .batch_normalization(relu=True, name='BatchNorm40')
             .conv(3, 3, 64, 1, 1, relu=False, name='Convolution41')
             .batch_normalization(name='BatchNorm41'))

        (self.feed('ReLU37', 
                   'BatchNorm41')
             .add(name='Eltwise19')
             .relu(name='ReLU39')
             .conv(3, 3, 64, 1, 1, relu=False, name='Convolution42')
             .batch_normalization(relu=True, name='BatchNorm42')
             .conv(3, 3, 64, 1, 1, relu=False, name='Convolution43')
             .batch_normalization(name='BatchNorm43'))

        (self.feed('ReLU39', 
                   'BatchNorm43')
             .add(name='Eltwise20')
             .relu(name='ReLU41')
             .conv(3, 3, 64, 1, 1, relu=False, name='Convolution44')
             .batch_normalization(relu=True, name='BatchNorm44')
             .conv(3, 3, 64, 1, 1, relu=False, name='Convolution45')
             .batch_normalization(name='BatchNorm45'))

        (self.feed('ReLU41', 
                   'BatchNorm45')
             .add(name='Eltwise21'))

        (self.feed('ReLU11')
             .conv(3, 3, 64, 4, 4, relu=False, name='Convolution_eltwise5')
             .batch_normalization(name='BatchNorm_Convolution_eltwise5'))

        (self.feed('ReLU27')
             .conv(3, 3, 64, 2, 2, relu=False, name='Convolution_eltwise13')
             .batch_normalization(name='BatchNorm_Convolution_eltwise13'))

        (self.feed('BatchNorm_Convolution_eltwise5', 
                   'BatchNorm_Convolution_eltwise13')
             .add(name='fuse1'))

        (self.feed('fuse1', 
                   'Eltwise21')
             .add(name='fuse2')
             .relu(name='ReLU49')
             .avg_pool(6, 6, 1, 1, padding='VALID', name='Pooling1')
             .fc(9, relu=False, name='InnerProduct1'))