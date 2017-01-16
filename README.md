# deep_class_aware_denoising
This is the TensorFlow code used to generate the results in https://arxiv.org/abs/1701.01698

We have provided code that can be used to reproduce most of the results in the paper. One may change the noise level, the model that is used, the data directory and whether the images are padded, by altering the flags inside d20_V2_denoise_test.py

We provide models for our class-agnostic models for noise levels of 25 and 50. Additional noise levels will be added at a later stage.

Training code is not provided at this stage but can easily be created by running the model.gen_train_op while feeding the training data.

The training/validation/test sets of PASCAL VOC 2010 are provided in the coresponding text files.
