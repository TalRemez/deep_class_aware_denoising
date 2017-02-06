# Deep Class-Aware Image Denoising
This is the TensorFlow code used to generate the results in https://arxiv.org/abs/1701.01698

We have provided code that can be used to reproduce most of the results in the paper. One may change the noise level, the model that is used, the data directory and whether the images are padded, by altering the flags of denoise_test.py. Unless explicitly stated the model that is used will be the one corresponding to the noise level selected.

For an example execute: denoise_test.py --noise_sigma=25

We provide models for our class-agnostic models for noise levels of 10, 15, 25, 35, 50 ,65, and 75.

Training code is not provided at this stage but can easily be created by running the model.gen_train_op while feeding training data.

The training/validation/test sets of PASCAL VOC 2010 are provided in the coresponding text files.

## TODO
* add training code
* add class-aware models
* add models for Poisson image denoising

