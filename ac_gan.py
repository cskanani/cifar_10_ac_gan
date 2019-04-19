import os
from collections import defaultdict
import pickle
from keras.layers import *
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.utils.generic_utils import Progbar
from keras import regularizers, optimizers
import keras.backend as K
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

np.random.seed(42)
class_num = 10
K.set_image_dim_ordering('th') #set number of channels as 1st dimension to make generator code more understandable

def build_generator(latent_size):
    inl = Input(shape=(latent_size,))
    x = Dense(384 * 4 * 4, activation='relu', kernel_initializer='glorot_normal', bias_initializer='Zeros')(inl)
    x = Reshape((384, 4, 4))(x)
    x = Conv2DTranspose(192, (5,5), strides=2, padding='same', activation='relu', kernel_initializer='glorot_normal', bias_initializer='Zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(96, (5,5), strides=2, padding='same', activation='relu',kernel_initializer='glorot_normal', bias_initializer='Zeros')(x)
    x = BatchNormalization()(x)
    out = Conv2DTranspose(3, (5,5), strides=2, padding='same', activation='tanh', kernel_initializer='glorot_normal', bias_initializer='Zeros')(x)
    gen = Model(inputs=inl, outputs=out)
    
    latent = Input(shape=(latent_size,))
    image_class = Input(shape=(1,), dtype='int32')
    cls = Flatten()(Embedding(10, latent_size, embeddings_initializer='glorot_normal')(image_class))
    h = multiply([latent, cls])

    fake_image = gen(h)
    return Model([latent, image_class], fake_image)


def build_discriminator():
    weight_decay = regularizers.l2(0.0005)
    inl = Input(shape=(3, 32, 32))
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=weight_decay)(inl)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=weight_decay)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=weight_decay)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=weight_decay)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=weight_decay)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=weight_decay)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=weight_decay)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=weight_decay)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=weight_decay)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=weight_decay)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=weight_decay)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=weight_decay)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=weight_decay)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu',  kernel_regularizer=weight_decay)(x)
    x = BatchNormalization()(x)
    out = Dropout(0.5)(x)
    model = Model(inputs=inl, outputs=out)
    image = Input(shape=(3, 32, 32))
    features = model(image)
    
    fake = Dense(1, activation='sigmoid', name='generation', kernel_initializer='glorot_normal', bias_initializer='Zeros')(features) #check if input is fake or real
    aux = Dense(class_num, activation='softmax', name='auxiliary', kernel_initializer='glorot_normal', bias_initializer='Zeros')(features) #class for the image

    return Model(image, [fake, aux])
        

def normalize(x_train,x_test):
    mean = np.mean(x_train,axis=(0,1,2,3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train-mean)/(std+1e-7)
    x_test = (x_test-mean)/(std+1e-7)
    return x_train, x_test, mean, std

def lr_scheduler(epoch):
    return 0.1 * (0.5 ** (epoch // 20))

        
if __name__ == '__main__':
    mx_val_acc = 0 #used for storing model with highest accuracy

    # batch and latent size taken from the paper
    epochs = 200
    batch_size = 100
    latent_size = 110
    
    #used SGD+momentum+nesterov as it has good generalization capacity than ADAM
    learning_rate = 0.1
    lr_decay = 1e-6
    lr_drop = 20    
    sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    
    #building discriminator
    discriminator = build_discriminator()
    discriminator.compile(optimizer=sgd, loss=['binary_crossentropy', 'categorical_crossentropy'])
    #discriminator.load_weights('best_discriminator.h5')
    
    #building generator
    generator = build_generator(latent_size)
    generator.load_weights('generator_1000.h5') #using pretrained weights
    
    #splitting and normalizing data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train,x_test,mean,std = normalize(x_train,x_test)
    pickle.dump({'x_test':x_test, 'y_test':y_test}, open('test_data_ac_gan_f.pkl','wb')) #storing normalized test data, it will be used for testing the model
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, stratify = y_train) #splitting data for validation
    nb_train, nb_val = x_train.shape[0], x_val.shape[0]

    #for storing the training and testing history
    train_loss = defaultdict(list)
    val_loss = defaultdict(list)
    val_acc = defaultdict(list)
    
    
    ##this code can be used for predictions
    #dct = pickle.load(open('test_data_ac_gan.pkl','rb'))
    #x_test = dct['x_test']
    #y_test = dct['y_test']
    #discriminator.load_weights('best_discriminator.h5')
    #y_pred = discriminator.predict(x_test, verbose=False)
    #print(classification_report(y_test, y_pred.argmax(-1).reshape(-1,)))
    #print(accuracy_score(y_test, y_pred.argmax(-1).reshape(-1,)))
    
    #training discriminator
    for epoch in range(epochs):
        learning_rate = lr_scheduler(epoch+1)
        K.set_value(discriminator.optimizer.lr, learning_rate)
        
        print('Epoch {} of {}'.format(epoch, epochs))
        nb_batches = nb_train//batch_size
        nb_batches = 2
        progress_bar = Progbar(target=nb_batches)

        epoch_disc_loss = []
        for index in range(nb_batches):
            progress_bar.update(index)
            
            #training on batch of original images
            x_real = x_train[index * batch_size:(index + 1) * batch_size]
            aux_y1 = to_categorical(y_train[index * batch_size:(index + 1) * batch_size].reshape(-1))
            y_real = np.random.uniform(0.7, 1.0, size=(batch_size,))
            epoch_disc_loss.append(discriminator.train_on_batch(x_real, [y_real, aux_y1]))
            
            #training on batch of generated images
            noise = np.random.normal(0, 0.5, (batch_size, latent_size))
            aux_y2 = to_categorical(np.random.randint(0, class_num, batch_size))
            x_fake = generator.predict([noise, aux_y2.argmax(-1).reshape((-1, 1))], verbose=0)
            x_fake = (x_fake-mean)/(std+1e-7) #normalizing the generated images
            y_fake = np.random.uniform(0.0, 0.3, size=(batch_size,))
            epoch_disc_loss.append(discriminator.train_on_batch(x_fake, [y_fake, aux_y2]))
        
        
        print('\nTesting for epoch {}:'.format(epoch))

        discriminator_val_loss = discriminator.evaluate(x_val, [np.array([1]*nb_val), to_categorical(y_val.reshape(-1))], verbose=False)
        
        y_pred = discriminator.predict(x_val, verbose=False)
        discriminator_val_acc = accuracy_score(y_val.reshape(-1), y_pred[1].argmax(-1).reshape(-1,))
        
        print('Val accuracy is :',discriminator_val_acc)
        if discriminator_val_acc > mx_val_acc:
            discriminator.save_weights('best_discriminator_f.h5')
            print('Saved weights')
            mx_val_acc = discriminator_val_acc

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        train_loss['discriminator'].append(discriminator_train_loss)
        val_loss['discriminator'].append(discriminator_val_loss)
        val_acc['discriminator'].append(discriminator_val_acc)

        print('-' * 65)
        print('discriminator loss(train):', train_loss['discriminator'][-1])
        print('discriminator loss(val):', val_loss['discriminator'][-1])
        print('discriminator accuracy(val):', discriminator_val_acc)
        
        #generating 100 randomm images, and storing them on drive
        noise = np.random.normal(0, 0.5, (100, latent_size))
        smpl_lbls = np.array([[i] * 10 for i in range(10)]).reshape(-1, 1)
        gen_imgs = generator.predict([noise, smpl_lbls]).transpose(0, 2, 3, 1)
        gen_imgs = np.asarray((gen_imgs * 127.5 + 127.5).astype(np.uint8))

        def vis_square(data, padsize=1, padval=0):
            n = int(np.ceil(np.sqrt(data.shape[0])))
            padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
            data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

            data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
            data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
            return data

        img = vis_square(gen_imgs)
        Image.fromarray(img).save('images_f/plot_epoch_{0:03d}_generated.png'.format(epoch))
        
        #storing training and testing history
        pickle.dump({'train': train_loss, 'val': val_loss, 'val_acc': val_acc}, open('acgan_history_f.pkl', 'wb'))
