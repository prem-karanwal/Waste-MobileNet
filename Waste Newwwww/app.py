import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.mobilenet import preprocess_input
from os.path import normpath, join, dirname
import ast

MODEL = None
MODEL_SERVING_FUNCTION = None

def full_path(filename: str) -> str:
    return normpath(join(dirname(__file__), filename))

def init() -> None:
    global MODEL, MODEL_SERVING_FUNCTION
    model_path = full_path("mobilenet_model")  
    MODEL = tf.saved_model.load(model_path)
    MODEL_SERVING_FUNCTION = MODEL.signatures['serving_default']

def run(input_data: dict) -> dict:
    if MODEL is None or MODEL_SERVING_FUNCTION is None:
        raise ValueError("Model is not loaded. Please call init() first.")

    img = load_img(input_data['image'], target_size=(224, 224))  
    img_array = preprocess_input(np.expand_dims(img, axis=0)) 

    
    predictions = MODEL_SERVING_FUNCTION(tf.convert_to_tensor(img_array, dtype=tf.float32))
    waste_pred = predictions['dense_1'].numpy()[0]  

    waste_types = ast.literal_eval(input_data['classifiers'][0])  
    index = np.argmax(waste_pred) 
    waste_label = waste_types[index]  
    accuracy = f"{waste_pred[index] * 100:.2f}"
    
    return {"accuracy": accuracy, "label": waste_label}

if __name__ == "__main__":
    init()
    
    input_data = {
        "image": "bot2.jpg",  
        "classifiers": ["['cardboard', 'glass', 'metal', 'paper','plastic', 'trash','biological','battery','shoes','clothes']"]  
    }
    
    prediction = run(input_data)
    print(f"Predicted label: {prediction['label']}, Accuracy: {prediction['accuracy']}%")
