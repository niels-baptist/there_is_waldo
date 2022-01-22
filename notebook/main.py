# THIS FILE IS NOT USED IN THE SOLUTION
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import SGD
import os

images_dir = "./images/256"
target_size = (256, 256)
target_dims = (256, 256, 3)
n_classes = 2
val_frac = 0.1
batch_size = 16

# , shear_range=0.2, zoom_range=0.2, horizontal_flip=True
data_gen = ImageDataGenerator(validation_split=val_frac)

train_generator = data_gen.flow_from_directory(
    images_dir, target_size=target_size, batch_size=batch_size, shuffle=True, class_mode='categorical', subset='training')
val_generator = data_gen.flow_from_directory(
    images_dir, target_size=target_size, batch_size=batch_size, class_mode='categorical', subset='validation')

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=target_dims))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
# 2 because we have cat and dog classes
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])

model.summary()

history = model.fit(train_generator, validation_data=val_generator, epochs=10)

labels = ['No waldo', 'Waldo']
waldos = ['1_1_1.jpg', '2_0_1.jpg', '2_1_0.jpg', '3_3_0.jpg', '4_0_2.jpg']

for filename in os.listdir('./images/test'):
    test_image = load_img(
        f'./images/test/{filename}', target_size=(256, 256))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    train_generator.class_indices

    print(f"Image: {filename}")

    label = labels[np.argmax(result[0])]

    if filename in waldos and label == "Waldo":
        print("Correct: Waldo")

    if filename not in waldos and label == "No waldo":
        print("Correct: No waldo")

    if filename in waldos and label != "Waldo":
        print("Wrong: Waldo not found")

    if filename not in waldos and label != "No waldo":
        print("Wrong: No waldo but found")
