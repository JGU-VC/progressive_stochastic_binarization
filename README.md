
# Progressive Stochastic Binarization of Deep Networks
This repository is the code for the paper: [Progressive Stochastic Binarization](https://arxiv.org/abs/1904.02205).


```
@article{corr/abs-1904-02205,
  title     = {Progressive Stochastic Binarization of Deep Networks},
  author    = {David Hartmann and Michael Wand},
  journal   = {CoRR},
  volume    = {abs/1904.02205},
  year      = {2019},
  url       = {http://arxiv.org/abs/1904.02205},
}
```



Setup
-----
1. Make sure you have Tensorflow v1.13.1 installed.
2. Install python requirements:
    ```
    pip install --user -r requirements.txt
    ```
3. Prepare the Imagenet-Dataset (as .tfrecords) as described in
    https://github.com/tensorflow/models/tree/master/official/resnet
4. Place the tfrecords of Imagenet in `./download/imagenet/`

**for the ResNet18-Tests**
1. Train a ResNet18 from official Tensorflow-Models
    https://github.com/tensorflow/models/tree/master/official/resnet
2. Place the Checkpoints in `./ckpts_imgn/resnet18_slim`

**for the Classification Models**
1. Run for every model from 
    https://github.com/qubvel/classification_models/tree/master/classification_models
   that you want to evaluate the download script. E.g. for resnet50v2 run:
   ```
   py py/download_and_convert_keras_model.py resnet50v2
   ```


Experiments
-----------
To run the experiments that produce the output to the tables of the paper run the scripts from the base directory.
For instance:
    ```
    sh experiments/test_attention.sh
    ```

General Useage
--------------
Please check the experiments for example scripts or check the optional arguments by
```
python bitnetwork.py --help
```
