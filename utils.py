import png
import numpy as np

def save_rgb_img(img,path,bit_depth):
	img = np.clip(np.array(img),0.0,255.0).astype(np.uint8)	
	f = open(path, 'wb')      # binary mode is important
	w = png.Writer(width=img.shape[1],
			height=img.shape[0],
			greyscale=False,
			bitdepth=bit_depth)
	w.write(f, img.reshape(-1, img.shape[1]*img.shape[2]).tolist())
	f.close()

def save_gray_img(img,path,bit_depth=8,scale=1):
	img = np.clip(np.array(img),0.0,255.0).astype(np.uint8)	
	f = open(path, 'wb')      # binary mode is important
	w = png.Writer(width=img.shape[1],
			height=img.shape[0],
			greyscale=True,
			bitdepth=bit_depth)
	w.write(f, (img/float(scale)).reshape(-1, img.shape[1]).tolist())
	f.close()


def rgb2y(rgb_img):
	return rgb_img[:,:,0:1]*0.299 + rgb_img[:,:,1:2]*0.587 + rgb_img[:,:,2:3]*0.114

def add_poiss_noise_image(img):
  sy,sx = img.shape
  lambda_flat = np.reshape(img,[-1,1]).astype(np.float32)
  noisy_flat = np.random.poisson(lam=lambda_flat)
  noisy = np.reshape(noisy_flat,[sy,sx])
  return(noisy.astype(np.float32))
