
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import glob
import os.path
import shutil



#getting all the Image in a File
img_path=os.path.join(os.getcwd(), 'Images/*.jpg')

file_name=[]

for image_file in glob.glob(img_path):
	if os.path.isfile(image_file) and image_file.endswith('.jpg'):
		file_name.append(image_file)
	else:
		print('There is no jpg image file to continue the Prediction')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



train_datagen = ImageDataGenerator(rescale = 1./255,
   									shear_range = 0.2,
   									zoom_range = 0.2,
   									horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('Dataset/Train',
												 target_size = (128, 128),
 												 batch_size = 32,
 												 class_mode = 'binary')


for given_file in file_name:
	test_image = image.load_img(given_file, target_size = (128, 128))
	# test_image.show()
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	classifier = load_model('my_model_transfer.h5')

	result = classifier.predict(test_image)

	#plt.imshow('test_image')
	training_set.class_indices
	if result[0][0] == 1:
		try:
			os.makedirs('Target_Gray')
		except FileExistsError:
			print('Folder Gray already had been created, Gray image will transfer to existing folder')
		prediction = 'gray image'
		dest_fold=os.path.join(os.getcwd(), 'Target_Gray')
		shutil.move(given_file, dest_fold)
	else:
		prediction = 'color image'
		try:
			os.makedirs('Target_Color')
		except FileExistsError:
			print('Folder Color already had been created, Color image will transfer to existing folder')
		prediction = 'Color image'
		dest_fold=os.path.join(os.getcwd(), 'Target_Color')
		shutil.move(given_file, dest_fold)		
