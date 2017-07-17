import csv
import cv2
import os
import click


batch_size = 64

def load_and_augment_data():
    samples = []
    corrections = [0, .2, -0.2]
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            for index in range(3):
                samples.append([line[index], float(line[3]) + corrections[index]])

    return samples

samples = load_and_augment_data()
make_path = lambda x: os.path.join('data/IMG', x.split('/')[-1])
read_image = lambda x: cv2.cvtColor(cv2.imread(make_path(x)), cv2.COLOR_BGR2RGB)

@click.group()
def cli():
    pass

@cli.command()
def show_stats():
    import matplotlib.pyplot as plt
    angles = []
    for sample in samples:
        angles.append(sample[1])
    plt.hist(angles, normed=True, bins=25)
    plt.ylabel('Count');
    plt.xlabel('Angle');
    plt.show()

def generator(samples, batch_size):
    from sklearn.utils import shuffle
    import numpy as np


    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image = read_image(batch_sample[0])
                angle = batch_sample[1]

                images.append(image)
                angles.append(angle)

                images.append(cv2.flip(image, 1))
                angles.append(-angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def make_model():
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Dropout
    from keras.layers import Conv2D, Cropping2D, MaxPooling2D
    
    
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Conv2D(3, (5, 5), strides=1, padding='same', activation='relu'))
    model.add(Conv2D(24, (5, 5), strides=2, padding='same', activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=2, padding='same', activation='relu'))
    model.add(Conv2D(48, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model

@cli.command()
def train_model():
    from sklearn.model_selection import train_test_split

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    model = make_model()

    history_object = model.fit_generator(train_generator,
            steps_per_epoch = len(train_samples) / batch_size,
            validation_data = validation_generator,
            validation_steps = len(validation_samples) / batch_size,
            epochs = 5,
            verbose = 1)

    model.save('model.h5')


@cli.command()
def draw_model():
    model = make_model()

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

if __name__ == '__main__':
    cli()
