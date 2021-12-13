import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

model = load_model('lenet_xray.h5')

labelmap = {
    '0':'ntweld',
    '1':'weld'
}

def load_lenet(img_arr):
	
	img_lenet = cv2.resize(img_arr,(64,64))
	img_lenet = img_lenet.astype('float')/255.0
	img_lenet = img_to_array(img_lenet)
	img_lenet = np.expand_dims(img_lenet, axis=0)

	return model.predict(img_lenet)[0]
	
def inference(img):
	res = load_lenet(img)
	print(res)
	return labelmap[str(np.argmax(res))]

def main():
	print(inference(cv2.imread('test.png')))

if __name__ == '__main__':
	main()

	
