# Dog-vs-Cat-Classification-using-CNN
🐶 Cat vs Dog Classifier 🐱  
A deep learning project using CNN to classify images of cats and dogs.  

##  Features  
- Trained on a dataset of cat and dog images  
- Uses Convolutional Neural Networks (CNN)  
- Implemented with TensorFlow and Keras  
- Supports image uploads for real-time classification  

##  Dataset  
The model is trained on a dataset of cat and dog images. Ensure your dataset is properly formatted before training.  

##  Requirements  
Install the required dependencies using:  
```bash
pip install tensorflow numpy pillow ipywidgets matplotlib
Run the following command to train the model:

history = model.fit(train_generator, validation_data=test_generator, epochs=10)
After training, accuracy and loss graphs can be visualized using:
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

Upload an image and use the trained model to classify:
predict_image("path_to_image.jpg")

