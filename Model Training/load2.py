import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

def load_and_predict(input_data):
    # Load the saved model
    model = load_model('sample_model.h5')
    
    # Create a scaler instance
    scaler = StandardScaler()
    
    # Scale the input data
    scaled_input = scaler.fit_transform(input_data)
    
    # Make predictions
    predictions = model.predict(scaled_input)
    
    return predictions

# Example usage
if __name__ == "__main__":
    # Example input data (68 features as per the model's input shape)
    # You should replace this with your actual input data
    example_input = np.random.rand(1, 68)  # Shape: (1, 68) for single prediction
    
    # Make prediction
    result = load_and_predict(example_input)
    
    # Print the prediction
    print("Predicted values:")
    print("Acceleration:", result[0][0])
    print("Braking:", result[0][1])
    print("Clutch:", result[0][2])
    print("Steering:", result[0][3])