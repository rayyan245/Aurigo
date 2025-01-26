import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
# Function to generate synthetic dataset with new features
def generate_synthetic_dataset(num_samples=1000):
    data = {
        'Height-to-Width Ratio': np.random.uniform(1, 10, num_samples),
        'Average Stress': np.random.uniform(0.1, 100, num_samples),
        'Average Strain': np.random.uniform(0.01, 5.0, num_samples),
        'Torque': np.random.uniform(10, 500, num_samples),
        'Vibration': np.random.uniform(0.1, 10, num_samples),
        'Condition': [random.choice(['Nicely built', 'Structurally inefficient']) for _ in range(num_samples)]
    }
    
    return pd.DataFrame(data)

def preprocess_data(df):
    # Separate features and target columns
    X = df.drop('Condition', axis=1)
    y_condition = df['Condition'].apply(lambda x: 1 if x == 'Nicely built' else 0)
    
    # Improvement suggestions based on realistic thresholds
    y_improvements = df[['Height-to-Width Ratio', 'Average Stress', 'Average Strain', 'Torque', 'Vibration']].apply(
        lambda x: [
            1 if x['Height-to-Width Ratio'] > 6 else 0,  # Height-to-width ratio > 5
            1 if x['Average Stress'] > 30 else 0,        # Stress > 30 MPa
            1 if x['Average Strain'] > 0.003 else 0,     # Strain > 0.3%
            1 if x['Torque'] > 200 else 0,               # Torque > 200 Nm
            1 if x['Vibration'] > 0.5 else 0             # Vibration > 0.5 cm/s
        ], axis=1)
    
    return X, y_condition, pd.DataFrame(y_improvements.tolist(), columns=[
        'Height-to-Width Ratio Improvement', 'Reinforce Materials', 'Reinforce Structural Design', 'Torque Adjustment', 'Vibration Control'
    ])

# Custom dataset class for PyTorch
class BuildingDataset(Dataset):
    def __init__(self, X, y_condition, y_improvements):
        # Ensure proper type conversion for X, y_condition, and y_improvements
        self.X = torch.tensor(X.astype(np.float32).values, dtype=torch.float32)
        self.y_condition = torch.tensor(y_condition.astype(np.int64).values, dtype=torch.long)
        self.y_improvements = torch.tensor(y_improvements.astype(np.float32).values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y_condition[index], self.y_improvements[index]

# Define the PyTorch model
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

# Generate synthetic dataset
df = generate_synthetic_dataset(1000)

# Preprocess the data
X, y_condition, y_improvements = preprocess_data(df)

# Train-test split
X_train, X_test, y_train_condition, y_test_condition, y_train_improvements, y_test_improvements = train_test_split(
    X, y_condition, y_improvements, test_size=0.2, random_state=42)

# Create custom dataset for training and test
train_dataset = BuildingDataset(X_train, y_train_condition, y_train_improvements)
test_dataset = BuildingDataset(X_test, y_test_condition, y_test_improvements)

# Create DataLoader for training and test
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
model = BuildingModel(X.shape[1])
criterion_condition = nn.CrossEntropyLoss()
criterion_improvements = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    model.train()
    for batch in train_loader:
        inputs, labels_condition, labels_improvements = batch
        optimizer.zero_grad()
        condition_out, improvements_out = model(inputs)
        
        loss_condition = criterion_condition(condition_out, labels_condition)
        loss_improvements = criterion_improvements(improvements_out, labels_improvements)
        
        loss = loss_condition + loss_improvements
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
predictions_condition = []
true_labels_condition = []
predictions_improvements = []
true_labels_improvements = []
with torch.no_grad():
    for batch in test_loader:
        inputs, labels_condition, labels_improvements = batch
        condition_out, improvements_out = model(inputs)
        
        _, predicted_condition = torch.max(condition_out, 1)
        predictions_condition.extend(predicted_condition.cpu().numpy())
        true_labels_condition.extend(labels_condition.cpu().numpy())
        
        predictions_improvements.extend((improvements_out > 0.5).cpu().numpy())
        true_labels_improvements.extend(labels_improvements.cpu().numpy())

# Calculate accuracy
accuracy_condition = np.mean(np.array(predictions_condition) == np.array(true_labels_condition))
print(f'Condition Accuracy: {accuracy_condition:.3f}')

# Calculate improvement accuracy (percentage of improvements correctly predicted)
accuracy_improvements = np.mean(np.array(predictions_improvements) == np.array(true_labels_improvements))
print(f'Improvements Accuracy: {accuracy_improvements:.3f}')

# Generate suggestions for structurally inefficient buildings in the test set
model.eval()
suggested_improvements = []

with torch.no_grad():
    for batch in test_loader:
        inputs, labels_condition, labels_improvements = batch
        condition_out, improvements_out = model(inputs)
        
        # Get predictions for the condition
        _, predicted_condition = torch.max(condition_out, 1)
        
        # Loop through each sample in the batch
        for i in range(len(predicted_condition)):
            if predicted_condition[i] == 0:  # Structurally inefficient
                improvements = []
                # Check for each improvement suggestion based on model's output (threshold > 0.5 for binary)
                if improvements_out[i, 0] > 0.5:
                    improvements.append("Reduce height-to-width ratio.")
                if improvements_out[i, 1] > 0.5:
                    improvements.append("Reinforce structural materials.")
                if improvements_out[i, 2] > 0.5:
                    improvements.append("Consider insulation for low temperatures.")
                if improvements_out[i, 3] > 0.5:
                    improvements.append("Improve drainage system.")
                if improvements_out[i, 4] > 0.5:
                    improvements.append("Implement earthquake-resistant features.")
                
                if not improvements:
                    improvements.append("No improvements needed.")
                
                suggested_improvements.append(improvements)

# Print suggested improvements for the test set
for idx, improvements in enumerate(suggested_improvements):
    print(f"Building {idx} Improvement Suggestions: {improvements}")

# Save the model (after training)
torch.save(model.state_dict(), 'building_model.pth')
