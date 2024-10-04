import numpy as np
import tensorflow as tf
import warnings
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL = None

def init() -> None:
    global MODEL
    model_path = "custom_mobilenetv2_waste_classification.h5" 
    MODEL = tf.keras.models.load_model(model_path)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

def run_single_image(image_path: str, classifiers: list) -> dict:
    if MODEL is None:
        raise ValueError("Model is not loaded. Please call init() first.")
    

    img = load_img(image_path, target_size=(96, 96))  
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = preprocess_input(img_array) 
    predictions = MODEL.predict(img_array)
    waste_pred = predictions[0]  
    index = np.argmax(waste_pred)  
    waste_label = classifiers[index]  
    
    return {"label": waste_label, "confidence": waste_pred[index] * 100}

if __name__ == "__main__":
    init()  
    
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash', 'biological', 'battery', 'shoes', 'clothes']
    
    image_path = "shoe.jpg"  

    prediction = run_single_image(image_path, class_names)
    
    print(f"Predicted label: {prediction['label']}")
    print(f"Confidence: {prediction['confidence']:.2f}%")
