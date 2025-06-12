import os
import zipfile
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 2: Unzip Dataset
zip_path = r"C:\Users\Devi Singh\Downloads\task4.zip"
extract_path = r"C:\Users\Devi Singh\Downloads\task4_extracted"

if not os.path.exists(extract_path):
    print("⏳ Extracting zip file...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Extraction completed!")
else:
    print("✅ Already extracted!")

# Step 3: Load Dataset
def load_data(data_dir, classes, img_size=(100, 100), max_per_class=10):
    X, y = [], []
    for label, folder in enumerate(classes):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            continue
        count = 0
        for file in os.listdir(folder_path):
            if count >= max_per_class:
                break
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                X.append(img.flatten())
                y.append(label)
                count += 1
    return np.array(X), np.array(y)

# Step 4: Define Paths and Classes
data_directory = os.path.join(extract_path, "leapGestRecog", "leapGestRecog", "00")
gesture_classes = ["01_palm", "02_l", "03_fist"]

# Step 5: Load Images
X, y = load_data(data_directory, gesture_classes)

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train SVM Classifier
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Step 8: Evaluate Model
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=gesture_classes))

# Step 9: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=gesture_classes, yticklabels=gesture_classes, cmap='Blues')
plt.title(f"Confusion Matrix\nAccuracy: {acc:.2f}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Step 10: Show Sample Predictions
def show_predictions(X_test, y_test, y_pred, class_names, img_shape=(100, 100), samples=5):
    plt.figure(figsize=(15, 3))
    indices = random.sample(range(len(X_test)), samples)
    for i, idx in enumerate(indices):
        img = X_test[idx].reshape(img_shape)
        true_label = class_names[y_test[idx]]
        pred_label = class_names[y_pred[idx]]
        plt.subplot(1, samples, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {true_label}\nPred: {pred_label}", 
                  color='green' if true_label == pred_label else 'red')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

show_predictions(X_test, y_test, y_pred, gesture_classes)