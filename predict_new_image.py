import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('Dataset/Train',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_image = image.load_img('Use_Case/Challenging Images/color/100BN55HNPF_30_20180223T051516319Z.jpg', target_size = (128, 128))
test_image.show()
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
classifier = load_model('my_model.h5')

result = classifier.predict(test_image)

#plt.imshow('test_image')
training_set.class_indices
if result[0][0] == 1:
    prediction = 'gray image'
    print(prediction)
else:
    prediction = 'color image'
    print(prediction)
