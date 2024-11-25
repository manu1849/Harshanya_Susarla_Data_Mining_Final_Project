# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, brier_score_loss
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv(r"https://raw.githubusercontent.com/manu1849/Harshanya_Susarla_Data_Mining_Final_Project/refs/heads/main/data%20.csv")
data.head(3)

summary_table = pd.DataFrame({
    'Column': data.columns,
    'Rows': [data[col].count() for col in data.columns],
    '% Missing Rows': [data[col].isnull().mean() * 100 for col in data.columns],
    'Data Type': data.dtypes.values,
    'First Row': data.iloc[0].values,
    'Unique Values': [data[col].nunique() for col in data.columns]
})

print(summary_table)

# Fill missing values with the mean of the respective column
data.fillna(data.mean(), inplace=True)

# Normalize the numerical features using MinMaxScaler
scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Transform the target variable 'num' for binary classification
data_scaled['num'] = data_scaled['num'].apply(lambda x: 1 if x > 0 else 0)

# Split the data into features (X) and target (y)
X = data_scaled.drop(columns=['num'])
y = data_scaled['num']

plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

num_cols = len(data.columns)
num_rows = (num_cols + 2) // 4

plt.figure(figsize=(15, 5 * num_rows))

for i, col in enumerate(data.columns):
    plt.subplot(num_rows, 4, i + 1)
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.show()

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print shapes of the datasets
print("\nShapes of the datasets:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

# Train a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf.predict(X_test)

# Evaluate the model
print("\nRandom Forest - Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("Random Forest - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# Reshape input data for LSTM (3D shape required)
X_train_dl = np.expand_dims(X_train, axis=2)
X_test_dl = np.expand_dims(X_test, axis=2)

# Convert target to categorical format
y_train_dl = to_categorical(y_train)
y_test_dl = to_categorical(y_test)

# Build LSTM model
lstm_model = Sequential([
    LSTM(64, input_shape=(X_train_dl.shape[1], X_train_dl.shape[2]), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')  # 2 classes: 0 and 1
])

# Compile the model
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("\nTraining LSTM Model...")
history = lstm_model.fit(X_train_dl, y_train_dl, validation_data=(X_test_dl, y_test_dl), epochs=20, batch_size=32)

# Evaluate the model
print("\nEvaluating LSTM Model...")
loss, accuracy = lstm_model.evaluate(X_test_dl, y_test_dl)
print(f"LSTM Model Accuracy: {accuracy * 100:.2f}%")

# Predict on the test set
y_pred_dl = lstm_model.predict(X_test_dl)
y_pred_dl_classes = np.argmax(y_pred_dl, axis=1)

print("\nLSTM - Classification Report:")
print(classification_report(y_test, y_pred_dl_classes))

print("LSTM - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dl_classes))

# Train a Support Vector Machine (SVM) model
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train, y_train)

# Predict on the test set
y_pred_svm = svm.predict(X_test)

# Evaluate the model
print("\nSVM - Classification Report:")
print(classification_report(y_test, y_pred_svm))

print("SVM - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

# Function to calculate confusion matrix and derived metrics
def calculate_metrics(y_true, y_pred, y_pred_prob=None):
    """
    Calculate and return a dictionary of performance metrics.
    """
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Manual Metrics
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    metrics = {
        "True Positives (TP)": tp,
        "True Negatives (TN)": tn,
        "False Positives (FP)": fp,
        "False Negatives (FN)": fn,
        "False Positive Rate (FPR)": fpr,
        "False Negative Rate (FNR)": fnr
    }

    # Advanced Metrics
    if y_pred_prob is not None:
        metrics["ROC AUC"] = roc_auc_score(y_true, y_pred_prob)
        metrics["Brier Score"] = brier_score_loss(y_true, y_pred_prob)

    return metrics

# Function to calculate ROC curve
def plot_roc_curve(y_true, y_pred_prob, model_name):
    """
    Plot ROC curve for a given model's predictions.
    """
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_true, y_pred_prob):.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.show()

# Get predictions and probabilities
y_pred_rf = rf.predict(X_test)
y_pred_rf_prob = rf.predict_proba(X_test)[:, 1]

# Calculate metrics
rf_metrics = calculate_metrics(y_test, y_pred_rf, y_pred_rf_prob)
print("\nRandom Forest - Metrics:")
for key, value in rf_metrics.items():
    print(f"{key}: {value}")

# Plot ROC Curve
plot_roc_curve(y_test, y_pred_rf_prob, "Random Forest")

# Get predictions and probabilities
y_pred_dl_classes = np.argmax(y_pred_dl, axis=1)  # Predicted classes
y_pred_dl_prob = y_pred_dl[:, 1]  # Probability of class 1

# Calculate metrics
lstm_metrics = calculate_metrics(y_test, y_pred_dl_classes, y_pred_dl_prob)
print("\nLSTM - Metrics:")
for key, value in lstm_metrics.items():
    print(f"{key}: {value}")

# Plot ROC Curve
plot_roc_curve(y_test, y_pred_dl_prob, "LSTM")

# Get predictions and probabilities
y_pred_svm = svm.predict(X_test)
y_pred_svm_prob = svm.predict_proba(X_test)[:, 1]  # Probability of class 1

# Calculate metrics
svm_metrics = calculate_metrics(y_test, y_pred_svm, y_pred_svm_prob)
print("\nSVM - Metrics:")
for key, value in svm_metrics.items():
    print(f"{key}: {value}")

# Plot ROC Curve
plot_roc_curve(y_test, y_pred_svm_prob, "SVM")

