import os
import cv2

import numpy as np

from sklearn.model_selection import train_test_split

dataset_dir = 'dataset'

def load_dataset(size_img):

	img_list = []
	label_list = []
	
	for i,img_folder in enumerate(os.listdir(dataset_dir)):
		print(img_folder,i)
		
		folder_path = os.path.join(dataset_dir,img_folder)
		
		for img_file in os.listdir(folder_path):

			img_path = os.path.join(folder_path,img_file)
			img = cv2.imread(img_path)

			#resize
			img_w, img_h = size_img
			img = cv2.resize(img, (img_w,img_h))

			#append
			img_list.append(img)
			label_list.append(np.array(i))

	
	img_list = np.array(img_list, dtype="float") / 255.0
	label_list = np.array(label_list)

	(xtrain, xtest, ytrain, ytest) = train_test_split(img_list, label_list, test_size=0.25, random_state=42)

	return xtrain,ytrain,xtest,ytest


			 
		
