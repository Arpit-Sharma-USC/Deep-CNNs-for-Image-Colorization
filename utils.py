import os
import numpy as np
import PIL.Image as Image
import theano.tensor as T
import lasagne.layers as layers
import cPickle as pickle
from cnn_utils import tile_raster_images


def save_image_from_array(arr, file_name = 'image.png', directory = 'plots'):

	
	if (arr.dtype != 'uint8'):
		arr = np.array(np.floor(arr * 256), dtype = 'uint8')


	if (len(arr.shape) == 4):
		arr = arr.reshape((arr.shape[1], arr.shape[2], arr.shape[3]))
	

	if (arr.shape[0] == 1):
		arr = np.vstack([arr, arr, arr])
	
	arr = np.transpose(arr, (1, 2, 0))
	img = Image.fromarray(arr)
	
	if (not os.path.exists(directory)):
		os.makedirs(directory)
	img.save(directory + '/' + file_name)
	
	
def get_greyscale(images, random_greyscale = False, rng = None, rgb = False):
	
	cr = 0.29
	cg = 0.6
	cb = 0.11
	
	if (random_greyscale):
		cr = rng.uniform(low = 0.1, high = 0.8)
		cg = rng.uniform(low = 0.1, high = 0.9 - cr)
		cb = 1 - cr - cg
		

	cnt = images.shape[0]
	h = images.shape[2]
	w = images.shape[3]
	
	res = cr * images[0:cnt, 0:1, 0:h, 0:w] + \
	      cg * images[0:cnt, 1:2, 0:h, 0:w] + \
  		  cb * images[0:cnt, 2:3, 0:h, 0:w]

	if (rgb):
		if (isinstance(images, np.ndarray)):
			res = np.concatenate((res, res, res), axis = 1)
		else:
			res = T.concatenate((res, res, res), axis = 1)
  
	return res
	
	
def print_samples(images, forward, model_name, epoch, suffix = '', columns = 1, directory = 'plots'):
	""

	where = directory + "/samples_" + model_name
	if (not os.path.exists(where)):
		os.makedirs(where)
	
	if (not isinstance(images, np.ndarray)):
		images = np.array(images.eval())


	n_images = images.shape[0]
	print "==> Printing %d images to %s" % (n_images, where)
	

	grey_images = get_greyscale(images, rgb = True)
	out_images = forward(images)

	all_images = np.array([]).reshape((0,) + images.shape[1:])
	for index in xrange(n_images):
		all_images = np.concatenate([all_images, images[index:(index + 1)]], axis = 0)
		all_images = np.concatenate([all_images, grey_images[index:(index + 1)]], axis = 0)
		all_images = np.concatenate([all_images, out_images[index:(index + 1)]], axis = 0)
		
	R = all_images[..., 0, ..., ...]
	G = all_images[..., 1, ..., ...]
	B = all_images[..., 2, ..., ...]
	
	all_images = tile_raster_images(
		(R, G, B, None), 
		(32, 32), 
		(-(-len(images) // columns), 3 * columns),
		tile_spacing = (1, 1)
	)
	image = Image.fromarray(all_images)
	
	image_name = "epoch %d" % epoch
	image.save(where + '/' + image_name + suffix + ".png")
	
	
def plot_filters(filters, model_name, epoch, suffix = '', max_num_filters = 100, columns = 1, repeat = 5, directory = 'plots'):
	
	where = directory + "/filters_" + model_name
	if (not os.path.exists(where)):
		os.makedirs(where)

	num_filters = min(filters.shape[0], max_num_filters)
	print "==> Printing %d images to %s" % (num_filters, where)
	
	filters = filters[0:num_filters]
	filters = filters.repeat(repeat, axis = 2).repeat(repeat, axis = 3)
	
	if (filters.shape[1] == 3):
		R = filters[..., 0, ..., ...]
		G = filters[..., 1, ..., ...]
		B = filters[..., 2, ..., ...]
	else:
		R = filters[..., 0, ..., ...]
		G = R
		B = R
	
	filters = tile_raster_images(
		(R, G, B, None), 
		filters.shape[2:], 
		(-(-filters.shape[0] // columns), columns),
		tile_spacing = (2, 2)
	)
	image = Image.fromarray(filters)
	
	image_name = "epoch %d" % epoch
	image.save(where + '/' + image_name + suffix + ".png")
	
	
def save_model(network, epoch, model_name, learning_rate = 0.0, directory = 'models'):
	params = layers.get_all_param_values(network)
	file_name = model_name + "-ep" + str(epoch) + ".pickle"
	file_path = directory + '/' + file_name
	print "==> Saving model to %s" % file_path
	
	if (not os.path.exists(directory)):
		os.makedirs(directory)
	
	with open(file_path, 'w') as save_file:
		pickle.dump(
			obj = {
				'params' : params,
				'epoch' : epoch,
				'learning_rate' : learning_rate,
			},
			file = save_file,
			protocol = -1
		)


def load_model(network, file_name, directory = 'models'):
	file_path = directory + '/' + file_name

	
	with open(file_path, 'r') as load_file:
		dict = pickle.load(load_file)
		layers.set_all_param_values(network, dict['params'])
		
		return {
			'epoch' : dict['epoch'],
			'learning_rate' : dict['learning_rate']
		}
		
