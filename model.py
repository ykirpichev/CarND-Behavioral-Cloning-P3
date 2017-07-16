import csv
import cv2
import os
import numpy as np

samples = []
batch_size = 32

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

make_path = lambda x: os.path.join('data/IMG', x.split('/')[-1])

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = make_path(batch_sample[0])
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])

                # create adjusted steering measurements for the side camera images
                correction = .2 # this is a parameter to tune
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                left_image = cv2.imread(make_path(batch_sample[1]))
                right_image = cv2.imread(make_path(batch_sample[2]))

                # add images and angles to data set
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)

                images.append(cv2.flip(center_image, 0))
                images.append(cv2.flip(left_image, 0))
                images.append(cv2.flip(right_image, 0))
                angles.append(-center_angle)
                angles.append(-left_angle)
                angles.append(-right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, Cropping2D, MaxPooling2D


model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))

model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(24, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(36, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(48, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
#history_object = model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, epochs=2, verbose=1)
history_object = model.fit_generator(train_generator,
        steps_per_epoch=len(train_samples)/batch_size,
        validation_data=validation_generator,
        validation_steps=len(validation_samples)/batch_size, 
        epochs=1,
        verbose=1)

model.save('model.h5')

#print(history_object.history.keys())
#
#import matplotlib.pyplot as plt
#### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
