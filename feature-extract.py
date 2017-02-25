import mxnet as mx
import numpy as np
import cv2
import os
from multiprocessing import Pool
#from sklearn import cross_validation
import joblib
from skimage import io, transform

def get_extractor():
    model = mx.model.FeedForward.load('model/Inception-7', 1, ctx=mx.cpu(),
                                      numpy_batch_size=1)

    internals = model.symbol.get_internals()
    fea_symbol = internals["global_pool_output"]

    # if you have GPU, then change ctx=mx.gpu()
    feature_extractor = mx.model.FeedForward(ctx=mx.cpu(), symbol=fea_symbol, numpy_batch_size=1, arg_params=model.arg_params, aux_params=model.aux_params, allow_extra_params=True)

    return feature_extractor



def PreprocessImage(path):
    # load image
    img = io.imread(path)
    print("Original Image Shape: ", img.shape)
    # we crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    # resize to 299, 299
    resized_img = transform.resize(crop_img, (299, 299))
    # convert to numpy.ndarray
    sample = np.asarray(resized_img) * 256
    # swap axes to make image from (299, 299, 3) to (3, 299, 299)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean
    normed_img = sample - 128.
    normed_img /= 128.

    return np.reshape(normed_img, (1, 3, 299, 299))

def extract_features():
    folders = sorted(os.listdir('train'))
    
    print(folders)
    for n, dir in enumerate(folders):
	folder = os.path.join('train', dir)
	paths = sorted([os.path.join(folder, fn) for fn in os.listdir(folder)])
	#print(paths[0])
	#pool = Pool(1)
	#img_samples = pool.map(PreprocessImage, paths)
	#for i in range(0,len(paths)):
	#img_samples = preprocess_image(paths[0])
	for i in range(0,len(paths)):
		samples=PreprocessImage(paths[i])		
		#samples = np.vstack(img_samples)
		model = get_extractor()
		global_pooling_feature = model.predict(samples)
		np.save(os.path.join('feats2', 'feats%s' % n + str(i)), global_pooling_feature)
extract_features()
