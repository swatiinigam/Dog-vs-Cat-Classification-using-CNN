#!/usr/bin/env python
# coding: utf-8

# In[20]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[21]:


import zipfile
import os

zip_path = r"C:\Users\Swati\Downloads\archive (2).zip"

extract_path = "/content/dogs_vs_cats_dataset"  # Update this with desired extract location

# Extract ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("✅ Dataset extracted successfully!")


# In[22]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = os.path.join(extract_path, "train")
test_dir = os.path.join(extract_path, "test")

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255, rotation_range=20, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load images
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(64, 64), batch_size=32, class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(64, 64), batch_size=32, class_mode='binary'
)

print("✅ Data loaded successfully!")


# In[23]:


import tensorflow as tf
from tensorflow.keras import layers, models

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary Classification
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("✅ CNN Model created successfully!")


# In[ ]:


history = model.fit(train_generator, validation_data=test_generator, epochs=10)

print("✅ Model training completed!")


# In[ ]:


model.summary()


# In[ ]:


model.save("cat_vs_dog_cnn.h5")
print("✅ Model saved successfully!")


# In[ ]:


get_ipython().system('pip install --upgrade ipywidgets traitlets')


# In[ ]:


import ipywidgets as widgets
widgets.FileUpload()


# In[ ]:


import numpy as np
import tensorflow as tf
import PIL.Image
import io
import ipywidgets as widgets
from tensorflow.keras.preprocessing import image
from IPython.display import display

# Load the trained model
model = tf.keras.models.load_model("cat_vs_dog_cnn.h5")

# Function to preprocess image and predict
def predict_image(uploaded_file):
    img = PIL.Image.open(uploaded_file).convert("RGB")
    img = img.resize((64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    prediction = model.predict(img_array)[0][0]
    result = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"

    display(img)
    print(f"Prediction: {result}")

# Create upload button
upload_btn = widgets.FileUpload(accept='image/*', multiple=False)

# Handle upload
def on_upload(change):
    uploaded_data = upload_btn.value  # Get uploaded file data

    if isinstance(uploaded_data, dict):  # Newer behavior (dictionary)
        filename = list(uploaded_data.keys())[0]  
        uploaded_file = uploaded_data[filename]['content']  

    elif isinstance(uploaded_data, tuple):  # Older behavior (tuple)
        uploaded_file = uploaded_data[0]['content']

    else:
        print("Unexpected file format:", type(uploaded_data))
        return

    predict_image(io.BytesIO(uploaded_file))

# Observe the upload button
upload_btn.observe(on_upload, names='value')

# Display the upload button
display(upload_btn)


# In[ ]:


train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Print final accuracy
print(f"Final Training Accuracy: {train_accuracy[-1] * 100:.2f}%")
print(f"Final Validation Accuracy: {val_accuracy[-1] * 100:.2f}%")


# In[ ]:


import matplotlib.pyplot as plt

# Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()


# In[ ]:




