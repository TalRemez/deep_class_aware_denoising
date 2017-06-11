import tensorflow as tf

class model:
	net = None
	train_op = None
	global_step = 0

def basic_denoise_model(data,params):
	print('-----------------------------------------------------')
	print('--------------------- Parameters --------------------')
	print('-----------------------------------------------------')
	print(params)
	print('-----------------------------------------------------')

	global_step = tf.Variable(0, name='global_step', trainable=False)

	#########################################################
	##	INPUTS
	#########################################################
	net = {}
	net['gt_img'] = data['gt_img']
	net['input_img'] = data['input_img']
	use_seg = False
	use_per_class_img = False
	print('******************************************')
	net['input'] = net['input_img']
	print('******************************************')
	prev_layer_name = 'input'

	#########################################################
	##	GENERATOR
	#########################################################
	print('Generator V2')
	add_generator_V2(net,net['input'],params)

	#########################################################
	##	ANALYSIS
	#########################################################
	net['diff_img'] = tf.abs(net['gt_img'] - net['gen_output'])
	
	net['error'] = tf.reduce_sum(input_tensor=net['diff_img'],
					reduction_indices=3,
					keep_dims=True)

	# remove the borders of the image
	net['diff_img'] = net['diff_img'][:,params['remove_border_size']:(-params['remove_border_size']),params['remove_border_size']:(-params['remove_border_size']),:]
	mse = tf.reduce_sum(tf.square(net['diff_img']))/tf.to_float(tf.size(net['diff_img']))
	psnr = 10.0*tf.log(1.0/mse)/tf.log(10.0)	

	m = model()
	m.net = net
	m.global_step = global_step
	m.mse = mse
	m.psnr = psnr
	return m

def add_generator_V2(net,input,params):
	prev_num_kernels =  1
	prev_layer_name = 'gen_input'
	net[prev_layer_name] = input
	with tf.variable_scope("generator"):
		for i in range(params['num_layers']):
			print('-----------------------------------------------------')
			print('----------------- Gen Layer %d ----------------------'%i)
			print('-----------------------------------------------------')
			print('input size '+str(net[prev_layer_name]))
			weights_name = 'W_%d'%i
			bias_name = 'b_%d'%i
			current_layer_name = 'layer_%d_out'%i

			if i==(params['num_layers']-1):
				num_kernels = 1
			else:

				num_kernels = params['num_kernels']

			if 'load_base_weights' in params and params['load_base_weights']==True:
				print('using value w_%d'%i)
				init_w_val = params['base_weights']['W_%d'%i]
				init_b_val = params['base_weights']['b_%d'%i]
			else:
				init_w_val = None
				init_b_val = None	

			net[weights_name] = weight_variable(	shape=[params['kernel_size'],
									params['kernel_size'],
									prev_num_kernels,
									num_kernels],
								name=weights_name,
								init_val=init_w_val)

			net[bias_name] = bias_variable([num_kernels],bias_name,init_val=init_b_val)
			print('weights tensor '+str(tf.shape(net[weights_name])))
			print('bias tensor '+str(tf.shape(net[bias_name])))

			net[current_layer_name] = tf.nn.conv2d(input=net[prev_layer_name],
					filter=net[weights_name],
					strides=[1,1,1,1],
					padding="SAME",
					use_cudnn_on_gpu=True)
	
			net[current_layer_name] = net[current_layer_name] + net[bias_name]

			if 'res' in params and params['res']==True or i==(params['num_layers']-1):
				net['gen_predicted_residual_%d'%i] = net[current_layer_name][:,:,:,0:1]
				
				if 'gen_predicted_residual' in net:
					net['gen_predicted_residual'] += net['gen_predicted_residual_%d'%i]
				else:
					net['gen_predicted_residual'] = net['gen_predicted_residual_%d'%i]
				net[current_layer_name] = net[current_layer_name][:,:,:,1:]
				prev_num_kernels = num_kernels-1
			else:
				prev_num_kernels = num_kernels

			if i<params['num_layers']-2:
				net[current_layer_name] = tf.nn.relu(net[current_layer_name])

			prev_layer_name = current_layer_name
			print('output size ' + str(net[prev_layer_name]))

		print('-----------------------------------------------------')
	
	net['gen_output'] = net['gen_predicted_residual']/10.0 + net['input_img']


def weight_variable(shape,name,init_val=None):
  if init_val==None:
	var = tf.get_variable(name,
				shape=shape,
				dtype=tf.float32,
		   		initializer=tf.contrib.layers.xavier_initializer_conv2d())
  else:
  	var = tf.Variable(init_val,dtype=tf.float32,name=name)
  return var

def bias_variable(shape,name,init_val=None):
  if init_val==None:
	b = tf.Variable(tf.ones(shape=shape,dtype=tf.float32)*0.01,	
			dtype=tf.float32,	
			name=name)
  else:
	b = tf.Variable(init_val,	
			dtype=tf.float32,	
			name=name)	
  return b

