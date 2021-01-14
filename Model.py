import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D,Conv2D, GlobalAveragePooling2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.layers import Input, BatchNormalization, Flatten
from keras import regularizers
import tensorflow as tf

basemodel = tf.keras.applications.ResNet50(
    include_top=False,weights="imagenet",
input_tensor=Input(shape=(128, 259, 3)))

for layer in basemodel.layers:
	layer.trainable = False

headmodel = basemodel.output

headmodel = AveragePooling2D(pool_size=(4, 4))(headmodel)
headmodel = Flatten(name="flatten")(headmodel)
headmodel = Dense(128, activation="relu")(headmodel)
headmodel = Dropout(0.5)(headmodel)
headmodel = Dense(8, activation="softmax",kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-3))(headmodel)
model = tf.keras.Model(inputs=basemodel.input, outputs=headmodel)


opt = tf.keras.optimizers.Adam(lr=1e-4)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


