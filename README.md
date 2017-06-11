# Deep Class-Aware Image Denoising
This page contains TensorFlow 0.11 and 1.0.1 (Python 2.7) code and models used to generate the results in https://arxiv.org/abs/1701.01698 and https://arxiv.org/pdf/1701.01687.pdf

![Alt text](/teaser.png?raw=true "Teaser")

We have provided code that can be used to reproduce most of the results in the paper. One may change the noise level, the model that is used, the data directory and whether the images are padded, by altering the flags of denoise_test.py. 

For an example for Gaussian noise with sigma of 25 execute: 
```
python denoise_test.py --noise_type=gaussian --noise_sigma=25 
```

For Poisson noise with peak value of 4 execute: 
```
python denoise_test.py --noise_type=poisson --noise_peak=4
```

We provide our class-agnostic models for gaussin noise with sigmas: 10, 15, 25, 35, 50 ,65, and 75. And Poisson noise with peak values of: 1,2,4,8, and 30.

Training code is not provided at this stage.

The training/validation/test sets of PASCAL VOC 2010 are provided in the coresponding text files.

## TODO
* add training code
* add class-aware models
