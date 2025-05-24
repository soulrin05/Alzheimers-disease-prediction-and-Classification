import os
import numpy as np
import cv2
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Configurations
DATASET_PATH = 'dataset/'  # Folder containing images categorized in subfolders (CN, MCI, AD, etc.)
IMG_SIZE = (64, 64)  # Smaller image size for lower RAM usage
TEST_SIZE = 0.2  # 80% training, 20% testing
K = 3  # Reduced number of neighbors for KNN to save memory
PCA_COMPONENTS = 100  # Reduce dimensions using PCA

# Function to Load and Preprocess Data
def load_data(dataset_path):
    X, y = [], []
    class_labels = os.listdir(dataset_path)
    
    for label in class_labels:
        class_dir = os.path.join(dataset_path, label)
        if not os.path.isdir(class_dir):
            continue
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            img = img.flatten()  # Convert image to 1D array
            X.append(img)
            y.append(label)
    
    return np.array(X), np.array(y)

print("[*] Loading dataset...")
X, y = load_data(DATASET_PATH)
print(f"Dataset loaded: {X.shape[0]} samples.")

# Encode Labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalize Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Apply PCA to reduce dimensions
pca = PCA(n_components=PCA_COMPONENTS)
X = pca.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

# Define Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),  # Reduce depth to save RAM
    "KNN": KNeighborsClassifier(n_neighbors=K)
}

# Train and Evaluate Each Model
trained_models = {}
for name, model in models.items():
    print(f"\n[*] Training {name}...")
    model.fit(X_train, y_train)
    
    # Evaluate Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")

    # Save Model
    model_filename = f"{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, model_filename)
    trained_models[name] = model_filename
    print(f"Model saved as: {model_filename}")

# Save Preprocessing Objects
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")
print("\nLabel encoder, scaler, and PCA saved.")

print("\nAll models trained and saved successfully!")
