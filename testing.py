import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

img_dir = './content/testing/test.jpg'

cnn = load_model('./models/cnn')

test_img = image.load_img(img_dir, target_size=(500, 500), color_mode='grayscale')

# Preprocess
processed = image.img_to_array(test_img)
processed = processed/255
processed = np.expand_dims(processed, axis=0)

# Predict
predictions = cnn.predict(processed)

if predictions >= 0.5:
    print(f'Pneumonia ({predictions[0][0] * 100}%)')
else:
    print(f'Normal ({predictions[0][0] * 100}%)')