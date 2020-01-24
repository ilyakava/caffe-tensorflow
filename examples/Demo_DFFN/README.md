# [Demo_DFFN Example](https://github.com/weiweisong415/Demo_DFFN)

## Regenerating models

This code works in the [onnx docker container](https://github.com/onnx/tutorials/blob/master/pytorch_caffe2_docker.md):

`docker run -it --rm onnx/onnx-docker:cpu /bin/bash`

Clone this repo and then run:

`python convert.py examples/Demo_DFFN/models/train_indian_pines.prototxt --code-output-path=examples/Demo_DFFN/models/train_indian_pines.py &> examples/Demo_DFFN/models/train_indian_pines.log`

Copy out of the docker container with:

`docker cp e12114an7ebh:/root/programs/caffe-tensorflow/examples/Demo_DFFN/models/* ./`

where the hash is the container id from `docker ps`

## Notes on .prototxt edits [from source](https://github.com/weiweisong415/Demo_DFFN/tree/master/prototxt_files)

At top, removed layers with `type: "HDF5Data"` and added the shape of the input, which is specified in [this matlab script](https://github.com/weiweisong415/Demo_DFFN/blob/master/generating_data.m)

input: "data"
input_shape {
  dim: bs=100
  dim: nchannel=3
  dim: spatial=23
  dim: spatial=23
}

At the bottom removed layers with `bottom: "label"`.
