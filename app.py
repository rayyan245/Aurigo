from flask import Flask, request, jsonify
import torch
import numpy as np
import torch.nn as nn

# Define the model architecture (must match the training script)
class BuildingModel(nn.Module):
    def __init__(self, input_dim):
        super(BuildingModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2_condition = nn.Linear(128, 2)  # Output for condition (2 classes)
        self.fc2_improvements = nn.Linear(128, 5)  # Output for improvements (5 binary suggestions)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        condition_out = self.fc2_condition(out)
        improvements_out = torch.sigmoid(self.fc2_improvements(out))  # Sigmoid for binary classification
        return condition_out, improvements_out

# Load the model
model = BuildingModel(5)  # 5 input features
model.load_state_dict(torch.load('building_model.pth'))  # Ensure this file exists in the same directory
model.eval()  # Set model to evaluation mode

# Initialize Flask app
app = Flask(__name__)

# Define preprocessing function
def preprocess_input(data):
    # Convert input list to a tensor
    input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return input_tensor

# Define a prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        input_data = request.json['features']  # Expecting 'features' key with a list of 5 values
        if len(input_data) != 5:
            return jsonify({'error': 'Invalid input. Expected 5 features.'}), 400

        # Preprocess the input
        input_tensor = preprocess_input(input_data)

        # Perform prediction
        with torch.no_grad():
            condition_out, improvements_out = model(input_tensor)

        # Decode condition prediction
        predicted_condition = torch.argmax(condition_out, dim=1).item()
        condition = 'Nicely built' if predicted_condition == 1 else 'Structurally inefficient'

        # Decode improvement suggestions
        improvement_suggestions = []
        if condition == 'Structurally inefficient':
            if improvements_out[0, 0] > 0.5:
                improvement_suggestions.append("Reduce height-to-width ratio.")
            if improvements_out[0, 1] > 0.5:
                improvement_suggestions.append("Reinforce structural materials.")
            if improvements_out[0, 2] > 0.5:
                improvement_suggestions.append("Reinforce structural design.")
            if improvements_out[0, 3] > 0.5:
                improvement_suggestions.append("Adjust torque on structural elements.")
            if improvements_out[0, 4] > 0.5:
                improvement_suggestions.append("Implement vibration control measures.")
        
        # Return result as JSON
        return jsonify({
            'condition': condition,
            'improvement_suggestions': improvement_suggestions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  # Accessible on local network
