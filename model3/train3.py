# ====== train3.py ======
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("model3/mediapipe_landmarks.csv")

# Pisahkan fitur dan label
X = df.drop("label", axis=1).values.astype("float32")
y = df["label"].values

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Simpan scaler
os.makedirs("model3", exist_ok=True)
with open("model3/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# One-hot encoding label
lb = LabelBinarizer()
y_encoded = lb.fit_transform(y)

# Simpan label encoder
with open("model3/label_binarizer.pkl", "wb") as f:
    pickle.dump(lb, f)

# Split stratified
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Encode dan ubah ke categorical shape
# y_train = tf.keras.utils.to_categorical(lb.transform(y_train_raw), num_classes=len(lb.classes_))
# y_test = tf.keras.utils.to_categorical(lb.transform(y_test_raw), num_classes=len(lb.classes_))

y_train = lb.transform(y_train_raw)
y_test = lb.transform(y_test_raw)

# MLP Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(lb.classes_), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    tf.keras.callbacks.ModelCheckpoint("model3/model3.h5", save_best_only=True, monitor='val_accuracy')
]

# Train
history = model.fit(
    X_train_raw, y_train,
    validation_data=(X_test_raw, y_test),
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)

# Evaluate
model.load_weights("model3/model3.h5")
y_pred = model.predict(X_test_raw)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Report
print("\n✅ Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=lb.classes_))
acc = np.mean(y_pred_classes == y_true)
print(f"\n✅ Akurasi Akhir: {round(acc * 100, 2)}%")

# Confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix(y_true, y_pred_classes), annot=True, fmt='d', cmap='Blues',
            xticklabels=lb.classes_, yticklabels=lb.classes_)
plt.title(f'Confusion Matrix (Accuracy: {round(acc * 100, 2)}%)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()