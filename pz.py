import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# Load the dataset
df = pd.read_csv(r"F:\Seizure detection\archive\Epileptic Seizure Recognition.csv")

# Clean up columns
df.drop(columns=['Unnamed', 'id'], inplace=True, errors='ignore')

# Extract features and labels
X = df.iloc[:, 1:179].values
y = df['y'].apply(lambda x: 1 if x == 1 else 0).values  # 1 = seizure, 0 = non-seizure

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# Neural Network Model
class SeizureNet(nn.Module):
    def __init__(self):
        super(SeizureNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(178, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# Initialize model
model = SeizureNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 200
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print(f"📊 Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor.to(device))
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.cpu().numpy()

accuracy = accuracy_score(y_test, predicted)
print("\n✅ Accuracy: {:.2f}%".format(accuracy * 100))
print("\n📄 Classification Report:\n", classification_report(y_test, predicted))
print("🧾 Confusion Matrix:\n", confusion_matrix(y_test, predicted))

# =============================
# 🔎 USER INPUT FOR PREDICTION
# =============================

print("\n📥 Enter 178 EEG values (comma separated, or press Enter to use a test sample):")
user_input = input()

if user_input.strip() == "":
    sample_input = X_test[0]  # use first test example
    actual_label = y_test[0]
    print("⚠️ No input given. Using sample test EEG data.")
else:
    try:
        values = list(map(float, user_input.strip().split(",")))
        assert len(values) == 178
        sample_input = scaler.transform([values])[0]
        actual_label = None
    except:
        print("❌ Invalid input. Please enter 178 comma-separated numbers.")
        exit()

# Convert to tensor and predict
model.eval()
with torch.no_grad():
    input_tensor = torch.tensor(sample_input, dtype=torch.float32).to(device)
    input_tensor = input_tensor.unsqueeze(0)  # add batch dimension
    output = model(input_tensor)
    _, prediction = torch.max(output, 1)
    prediction = prediction.item()

result_text = "🟥 Seizure Detected" if prediction == 1 else "🟩 No Seizure Detected"
print(f"\n🧠 Prediction Result: {result_text}")
if actual_label is not None:
    actual_text = "Seizure" if actual_label == 1 else "No Seizure"
    print(f"✅ Actual Label: {actual_text}")

