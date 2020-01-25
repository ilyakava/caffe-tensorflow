# [Demo_DFFN Example](https://github.com/weiweisong415/Demo_DFFN)

## Running DFFN

`sh scripts/train_indian_pines.sh`

Tested on Python 2.7.14 (Anaconda), tensorflow 1.10.1, cuda 9.0.176, cudnn-7.0 (8.0 might work too). Red Hat Enterprise Linux Workstation release 7.6 (Maipo). GeForce GTX TITAN X.

## Re-generating models

If you want to re-generate the included model code from prototxt to python, then
you will need a caffe installation.
This code works in the [onnx docker container](https://github.com/onnx/tutorials/blob/master/pytorch_caffe2_docker.md):

`docker run -it --rm onnx/onnx-docker:cpu /bin/bash`

Clone this repo and then run:

`python convert.py examples/Demo_DFFN/models/train_indian_pines.prototxt --code-output-path=examples/Demo_DFFN/models/train_indian_pines.py &> examples/Demo_DFFN/models/train_indian_pines.log`

Copy out of the docker container with:

`docker cp e12114an7ebh:/root/programs/caffe-tensorflow/examples/Demo_DFFN/models/* ./`

where the hash is the container id from `docker ps`

### Notes on .prototxt edits [from source](https://github.com/weiweisong415/Demo_DFFN/tree/master/prototxt_files)

Some minor edits to the original prototxt files are needed for the conversion to
specify the size info of the layers.

At top, removed layers with `type: "HDF5Data"` and added the shape of the input, which is specified in [this matlab script](https://github.com/weiweisong415/Demo_DFFN/blob/master/generating_data.m)

input: "data"
input_shape {
  dim: #bs
  dim: #nchannel
  dim: #spatial
  dim: #spatial
}

For indian pines #bs, #nchannel, #spatial are 100, 3, 25 as seen in the original [matlab script](https://github.com/weiweisong415/Demo_DFFN/blob/master/generating_data.m)

At the bottom removed layers with `bottom: "label"`.
