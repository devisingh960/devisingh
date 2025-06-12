import os
import zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# Step 1: Unzip the dataset if not already unzipped
zip_path = r'C:/Users/Devi Singh/Downloads/task3.zip'
extract_dir = r'C:/Users/Devi Singh/Downloads/task3_extracted'

if not os.path.exists(extract_dir):
    print("🔓 Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("✅ Extraction complete.")
else:
    print("📁 Dataset already extracted.")

# Step 2: Prepare data path
DATA_DIR = os.path.join(extract_dir, 'PetImages')
IMG_SIZE = 100
CATEGORIES = ['Dog', 'Cat']

def create_training_data():
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)
        class_num = CATEGORIES.index(category)

        for img_name in tqdm(os.listdir(path), desc=f'Loading {category}'):
            try:
                img_path = os.path.join(path, img_name)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_array is None:
                    continue
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                flattened_array = resized_array.flatten()
                training_data.append([flattened_array, class_num])
            except Exception:
                continue

    np.random.shuffle(training_data)
    return training_data

# Step 3: Load data
print("🔄 Creating training data...")
training_data = create_training_data()

X, y = [], []
for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 5: Train SVM model
print("⚙ Training SVM model...")
classifier = svm.SVC(kernel='linear', C=1)
classifier.fit(X_train, y_train)
print("✅ Model training complete.")

# Step 6: Evaluation
y_pred = classifier.predict(X_test)

print("\n📊 Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=CATEGORIES))

# Step 7: Pie Chart Only
counts = np.bincount(y_pred)
if len(counts) < 2:
    counts = np.append(counts, 0)

plt.figure(figsize=(5, 5))
plt.pie(counts, labels=CATEGORIES, autopct='%1.1f%%', colors=['orange', 'lightblue'])
plt.title("Predicted Class Distribution")
plt.axis('equal')  # Ensures pie is a circle
plt.show()