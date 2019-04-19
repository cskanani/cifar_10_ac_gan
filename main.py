from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras import optimizers
from keras import regularizers
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
import pickle

def MLP(input_shape):
    inl = Input(shape=input_shape)
    x = Flatten()(inl)
    x = BatchNormalization()(x)
    x = Dropout(.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(.5)(x)
    x = Dense(256, activation='relu')(x)
    out = Dense(10,activation='softmax')(x)
    return Model(inputs=inl, outputs=out)

def CNN(input_shape):
    inl = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu')(inl)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(.5)(x)
    x = Dense(256, activation='relu')(x)
    out = Dense(10, activation='softmax')(x)
    return Model(inputs=inl, outputs=out)

def VGG(input_shape):
    weight_decay = regularizers.l2(0.0005)
    inl = Input(shape=input_shape)
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
    x = Dropout(0.5)(x)
    out = Dense(10, activation='softmax')(x)
    return Model(inputs=inl, outputs=out)

def normalize(X_train,X_test):
    mean = np.mean(X_train,axis=(0,1,2,3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train, x_test = normalize(x_train, x_test)
y_train = to_categorical(y_train, 10)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, stratify = y_train)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

learning_rate = 0.1
lr_decay = 1e-6
lr_drop = 20
sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))
reduce_lr = LearningRateScheduler(lr_scheduler)



model = MLP((32,32,3))
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
model_checkpoint = ModelCheckpoint('mlp_200.h5', monitor='val_loss', save_best_only=True)
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=128), epochs=200, validation_data=(x_val, y_val), callbacks=[reduce_lr, model_checkpoint], verbose=2)
with open('mlp_200.hist', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

print('MLP:')
model.load_weights('mlp_200.h5')
y_pred = model.predict(x_test).argmax(-1)
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))



model = CNN((32,32,3))
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
model_checkpoint = ModelCheckpoint('cnn_200.h5', monitor='val_loss', save_best_only=True)
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=128), epochs=200, validation_data=(x_val, y_val), callbacks=[reduce_lr, model_checkpoint], verbose=2)
with open('cnn_200.hist', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

print('\nCNN:')
model.load_weights('cnn_200.h5')
y_pred = model.predict(x_test).argmax(-1)
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))



model = VGG((32,32,3))
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
model_checkpoint = ModelCheckpoint('vgg_200.h5', monitor='val_loss', save_best_only=True)
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=128), epochs=200, validation_data=(x_val, y_val), callbacks=[reduce_lr, model_checkpoint], verbose=2)
with open('vgg_200.hist', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

print('\nVGG:')
model.load_weights('vgg_200.h5')
y_pred = model.predict(x_test).argmax(-1)
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
