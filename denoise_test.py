from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os import listdir
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import re
from PIL import Image, ImageMath

import models
import utils

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('noise_type', None, 'Noise type, gaussian or poisson')
flags.DEFINE_float('noise_sigma', None, 'In the case of Gaussian noise, sigma 10,15,25,35,50,65,75')
flags.DEFINE_float('noise_peak', None, 'In the case of Poisson noise, the peak value of the image 1,2,4,8,30')
flags.DEFINE_string('data_dir', './images/','Directory with the testing data.')
flags.DEFINE_string('data_type', 'jpg','The test data image types.')
flags.DEFINE_string('log_dir', './results/','Directory to save results to.')
flags.DEFINE_boolean('pad_images',True,'Whether to pad the images before inputting them to the network. Change to True if you wish to have the output image size equal to the input image.')

###########################################
# Do not chnages the flags below this line
###########################################
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_integer('kernel_size', 3, '')
flags.DEFINE_integer('num_kernels', 64, '')
flags.DEFINE_integer('num_layers', 20, '')
flags.DEFINE_boolean('res',True,'')
flags.DEFINE_integer('remove_border_size',20,'')


def run_testing():
  """Test denoising model."""
  params = {}
  params['kernel_size'] = FLAGS.kernel_size
  params['num_kernels'] = FLAGS.num_kernels
  params['num_layers'] = FLAGS.num_layers
  params['res'] = FLAGS.res
  params['remove_border_size'] = FLAGS.remove_border_size

  if FLAGS.noise_type == 'gaussian':
	if(FLAGS.noise_sigma is None):
		raise ValueError('please set noise sigma value')
	FLAGS.log_dir += 'gaussian_sigma_%d/'%(FLAGS.noise_sigma)	
	ckpt = './models/model_sigma_%d.ckpt'%FLAGS.noise_sigma
  elif FLAGS.noise_type == 'poisson':
	if(FLAGS.noise_peak is None):
		raise ValueError('please set noise peak value')
	FLAGS.log_dir += 'poisson_peak_%d/'%(FLAGS.noise_peak)
	ckpt = './models/model_peak_%d.ckpt'%FLAGS.noise_peak
  else:
	print('Noise type not supported!!!')
	return

  if not os.path.isdir(FLAGS.log_dir):
  	os.makedirs(FLAGS.log_dir)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():

    gt_img = tf.placeholder(tf.float32,shape=(FLAGS.batch_size,None,None,1))
    noisy_img = tf.placeholder(tf.float32,shape=(FLAGS.batch_size,None,None,1))

    data = {}
    data['gt_img'] = gt_img
    data['input_img'] = noisy_img

    # Pad the arrays so we can crop them later
    if FLAGS.pad_images:
    	padding = [[0,0],[params['remove_border_size'],params['remove_border_size']],[params['remove_border_size'],params['remove_border_size']],[0,0]]
    	data['gt_img'] = tf.pad(data['gt_img'],padding , "SYMMETRIC")
    	data['input_img'] = tf.pad(data['input_img'],padding , "SYMMETRIC")

    # Build a Graph that computes predictions from the inference model.
    model = models.basic_denoise_model(data=data,params=params)

    # The op for initializing the variables.
    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())

    # Create a session for running operations in the Graph.
    sess = tf.Session()

    saver = tf.train.Saver()
  
    # Initialize the variables
    print('restoring model '+ckpt)
    sess.run(init_op)
    saver.restore(sess, ckpt)

    path = FLAGS.data_dir
    print('Looking for images in '+ path)
    img_list = [f for f in listdir(path) if re.search(r'[.]*\.%s$'%FLAGS.data_type,f)]
    print('found %d files'%len(img_list))

    try:
	mse_list = []
	psnr_list = []
	img_name_list = []

	for img in img_list:

		
	  	# Get the data.
		print('-------------------------------------------------------')		
		
		path = FLAGS.data_dir + '/' + img
		print(path)
		rgb_img = Image.open(path)
		rgb_img = np.ndarray((rgb_img.size[1], rgb_img.size[0], 3), 'u1', rgb_img.tobytes())
		rgb_img = rgb_img.astype(np.float32) 
  		y_img = utils.rgb2y(rgb_img)

		gt = y_img
		gt = gt[:,:,0]
		

		if FLAGS.noise_type == 'gaussian':	
			gt = gt.astype(np.float32) * (1.0 / 255.0) - 0.5		
			noisy = gt + np.random.normal(size=gt.shape)*float(FLAGS.noise_sigma)/255.0
		else:
			max_val = np.amax(np.amax(gt))
			gt = gt.astype(np.float32) * (1.0 / float(max_val)) - 0.5	
			img_peak = (0.5+gt)*float(FLAGS.noise_peak)
    			noisy = utils.add_poiss_noise_image(img_peak).astype(np.float32)
    			noisy= (noisy/float(FLAGS.noise_peak))-0.5

		gt = np.expand_dims(a=gt, axis=2)
		gt = np.expand_dims(a=gt, axis=0)
		noisy = np.expand_dims(a=noisy, axis=2)
		noisy = np.expand_dims(a=noisy, axis=0)

		feed_dict={gt_img:gt,noisy_img:noisy}

		start_time = time.time()
	
		# Run the network
		(gt_image,
		input_image,
		output_image,
		mse,
		psnr) = sess.run([model.net['gt_img'],
					model.net['input_img'],
					model.net['gen_output'],
					model.mse,
					model.psnr
				],feed_dict=feed_dict)

		psnr_list.append(psnr)
		mse_list.append(mse)	
		duration = time.time() - start_time
		print('PSNR=%f %f[sec]'%(psnr,duration))
		
		input_image = np.array(np.clip((input_image+0.5)*255.0,0,255),dtype=np.uint8)	
		output_image = np.array(np.clip((output_image+0.5)*255.0,0,255),dtype=np.uint8)

		img_name = img[0:-(len(FLAGS.data_type)+1)]

		utils.save_gray_img(img=input_image[0,:,:,:],
					path=FLAGS.log_dir + '%s_noisy.png'%img_name,
					bit_depth=8)	
		utils.save_gray_img(img=output_image[0,:,:,:],
					path=FLAGS.log_dir + '%s_denoised.png'%img_name,
					bit_depth=8)	
		sio.savemat(FLAGS.log_dir + img_name + '.mat',{'gt_img':gt_image,'input_img':input_image,'denoised_img':output_image,'psnr':psnr,'mse':mse})	
	print('------------------ Overall performance ---------------------')		
	print('mean PSNR=%f'%np.mean(psnr_list))
	print('mean MSE=%f'%np.mean(mse_list))        

    except tf.errors.OutOfRangeError:
      print('Error')
    finally:
	pass

 
def main(_):
  run_testing()


if __name__ == '__main__':
  tf.app.run()
