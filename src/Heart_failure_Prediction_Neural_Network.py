import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, f1_score, classification_report
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# Set random seed for reproducibility
np.random.seed(42)

# Read the data
try:
    data_df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    print("Data loaded successfully!")
    print("\nDataset Info:")
    data_df.info()
except FileNotFoundError:
    print("Error: Dataset file not found!")
    exit()

# First visualization - Death Event Distribution
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
cols = ["#00FF00", "#FF0000"]
ax = sns.countplot(x=data_df["DEATH_EVENT"], palette=cols)
ax.set_title("Patient Survival Distribution", pad=15, fontsize=12, fontweight='bold')
ax.set_xlabel("Death Event", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.set_xticklabels(['Survivors', 'Deceased'])
for container in ax.containers:
    ax.bar_label(container, padding=3, fontsize=11, fontweight='bold')
sns.despine()
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data_df.corr(), cmap="BuGn", annot=True, fmt='.2f')
plt.title("Correlation Heatmap", pad=15, fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# Feature Distribution Plots
features = ["age", "creatinine_phosphokinase", "ejection_fraction", 
           "platelets", "serum_creatinine", "serum_sodium", "time"]
for feature in features:
    plt.figure(figsize=(10, 7))
    sns.boxenplot(x=data_df["DEATH_EVENT"], y=data_df[feature], palette=cols)
    sns.swarmplot(x=data_df["DEATH_EVENT"], y=data_df[feature], 
                  color="black", alpha=0.5, size=4)
    plt.title(f"Distribution of {feature.replace('_', ' ').title()}", 
              fontsize=12, fontweight='bold')
    plt.xlabel("Death Event (0=Survived, 1=Deceased)", fontsize=10)
    plt.ylabel(feature.replace('_', ' ').title(), fontsize=10)
    plt.tight_layout()
    plt.show()

# Data Preprocessing
X = data_df.drop("DEATH_EVENT", axis=1)
y = data_df["DEATH_EVENT"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                    test_size=0.3, 
                                                    random_state=42, 
                                                    stratify=y)

# SVM Model
print("\nTraining SVM Model...")
svm_model = svm.SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

print("\nSVM Model Performance:")
print(classification_report(y_test, svm_pred))

#ANN
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, 
    patience=10,
    restore_best_weights=True
)

# Get input shape from training data
input_shape = X_train.shape[1]

# Create model with Input layer
model = Sequential([
    Input(shape=(input_shape,)),
    Dense(16, kernel_initializer="uniform", activation='relu'),
    Dense(8, kernel_initializer="uniform", activation='relu'),
    Dropout(0.25),
    Dense(8, kernel_initializer="uniform", activation='relu'),
    Dropout(0.5),
    Dense(1, kernel_initializer="uniform", activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

history = model.fit(X_train, y_train, batch_size=25, epochs=100, callbacks=[early_stopping], validation_split=0.25)

history_df = pd.DataFrame(history.history)
plt.plot(history_df.loc[:,['loss']],label = "Training Loss")
plt.plot(history_df.loc[:,['val_loss']],label = "Validation Loss")
plt.legend()
plt.show()

plt.plot(history_df.loc[:,['accuracy']],label = "Training accuracy")
plt.plot(history_df.loc[:,['val_accuracy']],label = "Validation accuracy")
plt.legend()
plt.show()

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
print(classification_report(y_test, y_pred))