from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import os.path

num_classes = 2
resnet_weights_path = os.path.join(os.getcwd(), 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False

# Compile the model
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# Fit the model
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224


data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   horizontal_flip=True,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2)

# Removing the line for the process of data augmengtattion
# data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = data_generator_with_aug.flow_from_directory(
        os.path.join(os.getcwd(), 'Dataset/Train'),
        target_size=(image_size, image_size),
        batch_size=20,
        class_mode='categorical')


data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = data_generator_no_aug.flow_from_directory(
        os.path.join(os.getcwd(), 'Dataset/Test'),
        target_size=(image_size, image_size),
        class_mode='categorical')

my_new_model.fit_generator(
        train_generator,
        steps_per_epoch=22,
        epochs=1, #--10
        validation_data=validation_generator,
        validation_steps=3)

from keras.models import load_model

my_new_model.save('my_model_transfer.h5')

# returns a compiled model
# identical to the previous one
# model = tf.keras.models.load_model('my_model_transfer.h5')