import numpy as np
from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight

train_dir = './content/dataset/cnn/data/train'
test_dir = './content/dataset/cnn/data/test'
val_dir = './content/dataset/cnn/data/val'

dims = 500

# Augment Data
gen_img = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
gen_img_test = ImageDataGenerator(rescale=1./255)

# Load augmented data
train = gen_img.flow_from_directory(train_dir, target_size=(dims, dims), color_mode='grayscale', class_mode='binary', batch_size=16)
test = gen_img_test.flow_from_directory(test_dir, target_size=(dims, dims), color_mode='grayscale', class_mode='binary', batch_size=16, shuffle=False)
valid = gen_img_test.flow_from_directory(val_dir, target_size=(dims, dims), color_mode='grayscale', class_mode='binary', batch_size=16)

# Create cnn
cnn = Sequential()

cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(dims, dims, 1)))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(dims, dims, 1)))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(dims, dims, 1)))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Conv2D(64, (3, 3), activation='relu', input_shape=(dims, dims, 1)))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Conv2D(64, (3, 3), activation='relu', input_shape=(dims, dims, 1)))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Flatten())

cnn.add(Dense(activation='relu', units=128))
cnn.add(Dense(activation='relu', units=64))
cnn.add(Dense(activation='sigmoid', units=1))

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callback list
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=3)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.3, min_lr=0.000001)

cb_list = [early_stopping, learning_rate_reduction]

# Weights
weights = compute_class_weight('balanced', np.unique(train.classes), train.classes)
cw = dict(zip(np.unique(train.classes), weights))

# Train
cnn.fit(train, epochs=25, validation_data=valid, class_weight=cw, callbacks=cb_list)

# Test accuracy
accuracy = cnn.evaluate(test)
print('Test accuracy:', accuracy[1]*100, '%')

# Predict
predictions = cnn.predict(test, verbose=1)

pred_mx = predictions.copy()
pred_mx[pred_mx <= 0.5] = 0
pred_mx[pred_mx > 0.5] = 1

print(classification_report(y_true=test.classes, y_pred=pred_mx, target_names=['NORMAL','PNEUMONIA']))