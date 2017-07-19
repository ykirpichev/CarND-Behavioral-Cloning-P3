import csv
import cv2
import os
import click


batch_size = 64

def load_and_augment_data():
    import numpy as np

    image_paths = []
    angles = []
    corrections = [0, 0.15, -0.15]
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            for index in range(3):
                image_paths.append(line[index])
                angles.append(float(line[3]) + corrections[index])
              

    num_bins = 25
    print(len(angles))
    avg_samples_per_bin = len(angles) / num_bins
    hist, bins = np.histogram(angles, num_bins)

    keep_probs = []
    target = avg_samples_per_bin# * 0.5

    for i in range(num_bins):
        if hist[i] < target:
            keep_probs.append(1.)
        else:
            print(i, hist[i], bins[i], 1./(float(hist[i])/target))

            if abs(bins[i]) < 0.1:
                keep_probs.append(1.5/(float(hist[i])/target))
            else:
                keep_probs.append(1./(float(hist[i])/target))

    remove_list = []
    for i in range(len(angles)):
        for j in range(num_bins):
            if angles[i] >= bins[j] and angles[i] < bins[j+1]:
                # delete from X and y with probability 1 - keep_probs[j]
                if np.random.rand() > keep_probs[j]:
                    remove_list.append(i)
    image_paths = np.delete(image_paths, remove_list, axis=0)
    angles = np.delete(angles, remove_list)
    res = []
    for x, y in zip(image_paths, angles):
        res.append((x,y))
    return res

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

def make_small_model():
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Dropout
    from keras.layers import Conv2D, Cropping2D, MaxPooling2D
    
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Conv2D(3, (5, 5), strides=1, padding='same', activation='relu'))
    model.add(Conv2D(8, (3, 3), strides=2, padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), strides=2, padding='same', activation='relu'))
    model.add(Conv2D(24, (3, 3), strides=2, padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), strides=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
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

    print(history_object.history.keys())
    print('Loss')
    print(history_object.history['loss'])
    print('Validation Loss')
    print(history_object.history['val_loss'])

@cli.command()
def train_small_model():
    from sklearn.model_selection import train_test_split

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    model = make_small_model()

    history_object = model.fit_generator(train_generator,
            steps_per_epoch = len(train_samples) / batch_size,
            validation_data = validation_generator,
            validation_steps = len(validation_samples) / batch_size,
            epochs = 3,
            verbose = 1)

    model.save('model_small.h5')

    print(history_object.history.keys())
    print('Loss')
    print(history_object.history['loss'])
    print('Validation Loss')
    print(history_object.history['val_loss'])

@cli.command()
def draw_model():
    model = make_model()
    from keras.utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True)


@cli.command()
def draw_random_images():
    import matplotlib.pyplot as plt

    for i in range(3):
        image = read_image(samples[i][0])
        angle = samples[i][1]
        print(angle)
        plt.figure(i)
        plt.imshow(image)
    plt.show()
    plt.waitforbuttonpress()


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
