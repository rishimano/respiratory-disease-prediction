#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pandas
import numpy as numpy
import matplotlib.pyplot as pyplot
import os
import librosa
import librosa.display
import tensorflow as tensorflow
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import xgboost as xgboost
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
import joblib
import seaborn as seaborn
from tqdm.notebook import tqdm
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV

# Define file paths
AUDIO_DIRECTORY = 'dissertation/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/'
DEMOGRAPHIC_FILE = 'dissertation/demographic_info.txt'
DIAGNOSIS_FILE = 'dissertation/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv'

# Load demographic data
demographic_df = pandas.read_csv(DEMOGRAPHIC_FILE, names=['PatientID', 'Age', 'Gender', 'BMI', 'BodyWeight', 'BodyHeight'], delimiter=' ')

# Load diagnosis data
diagnosis_df = pandas.read_csv(DIAGNOSIS_FILE, names=['PatientID', 'Diagnosis'])

# Combine demographic and diagnosis data
combined_df = demographic_df.join(diagnosis_df.set_index('PatientID'), on='PatientID', how='left')

# Show diagnosis distribution
print(combined_df['Diagnosis'].value_counts())

# Function to load and process annotation files
def extract_annotation_data(file_name, root):
    tokens = file_name.split('_')
    recording_info = pandas.DataFrame(data=[tokens], columns=['PatientID', 'RecordingID', 'ChestLocation', 'AcquisitionMode', 'RecordingEquipment'])
    annotations = pandas.read_csv(os.path.join(root, file_name + '.txt'), names=['Start', 'End', 'Crackles', 'Wheezes'], delimiter='\t')
    return recording_info, annotations


def extract_features(audio_file, mode='mfcc'):
    sr_new = 16000  # Resample to 16kHz
    audio_data, sr = librosa.load(audio_file, sr=sr_new)
    max_len = 5 * sr_new  # Limit to 5 seconds

    # Padding or truncating audio
    if len(audio_data) < max_len:
        audio_data = numpy.pad(audio_data, (0, max_len - len(audio_data)))
    else:
        audio_data = audio_data[:max_len]
    
    if mode == 'mfcc':
        feature = librosa.feature.mfcc(y=audio_data, sr=sr_new)
    elif mode == 'log_mel':
        feature = librosa.feature.melspectrogram(y=audio_data, sr=sr_new, n_mels=128, fmax=8000)
        feature = librosa.power_to_db(feature, ref=numpy.max)
    
    return feature


# Extract features from the audio files
audio_data_list = []
labels = []

for file_name in tqdm(os.listdir(AUDIO_DIRECTORY)):
    if '.wav' in file_name:
        diagnosis = combined_df[combined_df['PatientID'] == int(file_name.split('_')[0])]['Diagnosis'].values[0]
        audio_file_path = os.path.join(AUDIO_DIRECTORY, file_name)
        features = extract_features(audio_file_path, mode='mfcc')
        audio_data_list.append(features)
        labels.append(diagnosis)

# Summary statistics
print(combined_df.describe())


# Histograms for demographic features
combined_df.hist(column=['Age', 'BMI', 'BodyWeight', 'BodyHeight'], bins=20, figsize=(10,8))
pyplot.suptitle('Histograms for Demographic Features')
pyplot.show()

# Convert to numpy arrays
preprocessed_audio = numpy.array(audio_data_list)
labels = numpy.array(labels)

# Reshape the audio data for model input
preprocessed_audio = preprocessed_audio.reshape((-1, 20, 157, 1))  # Reshaped for Conv2D input

# One-hot encoding of labels
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)
labels_one_hot = to_categorical(labels_encoded, num_classes=8)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(preprocessed_audio, labels_one_hot, test_size=0.2, random_state=42)


# If you want to plot MFCC distributions
pyplot.figure(figsize=(10,6))
pyplot.hist(preprocessed_audio.flatten(), bins=50)
pyplot.title('Distribution of MFCC Features')
pyplot.xlabel('MFCC Coefficients')
pyplot.ylabel('Frequency')
pyplot.show()


# Perform one-hot encoding for both 'Gender' and 'Diagnosis' columns
combined_df_encoded = pandas.get_dummies(combined_df, columns=['Gender', 'Diagnosis'], drop_first=True)
combined_df_encoded = combined_df_encoded.rename(columns={'Gender_1': 'Gender_F'})
combined_df_encoded = combined_df_encoded.rename(columns={'Gender_2': 'Gender_M'})

# Now, calculate the correlation matrix
correlation_matrix = combined_df_encoded.corr()

# Plot the heatmap
pyplot.figure(figsize=(12, 8))
seaborn.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
pyplot.title('Correlation Heatmap')
pyplot.show()

# Boxplot for demographic features
pyplot.figure(figsize=(10,6))
seaborn.boxplot(data=combined_df[['Age', 'BMI', 'BodyWeight', 'BodyHeight']])
pyplot.title('Boxplot for Demographic Features')
pyplot.show()


# Boxplot for demographic data grouped by Diagnosis
fig, axs = pyplot.subplots(2, 2, figsize=(12, 8))

# Boxplot for Age by Diagnosis
seaborn.boxplot(x='Diagnosis', y='Age', data=combined_df, ax=axs[0, 0])
axs[0, 0].set_title('Age by Diagnosis')
axs[0, 0].set_xticklabels(axs[0, 0].get_xticklabels(), rotation=45)

# Boxplot for BodyWeight by Diagnosis
seaborn.boxplot(x='Diagnosis', y='BodyWeight', data=combined_df, ax=axs[0, 1])
axs[0, 1].set_title('BodyWeight by Diagnosis')
axs[0, 1].set_xticklabels(axs[0, 1].get_xticklabels(), rotation=45)

# Boxplot for BodyHeight by Diagnosis
seaborn.boxplot(x='Diagnosis', y='BodyHeight', data=combined_df, ax=axs[1, 0])
axs[1, 0].set_title('BodyHeight by Diagnosis')
axs[1, 0].set_xticklabels(axs[1, 0].get_xticklabels(), rotation=45)

# Boxplot for BMI by Diagnosis
seaborn.boxplot(x='Diagnosis', y='BMI', data=combined_df, ax=axs[1, 1])
axs[1, 1].set_title('BMI by Diagnosis')
axs[1, 1].set_xticklabels(axs[1, 1].get_xticklabels(), rotation=45)

pyplot.tight_layout()
pyplot.show()


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(preprocessed_audio, labels_one_hot, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(16, kernel_size=2, input_shape=(20, 157, 1), activation='relu'),
    MaxPooling2D(pool_size=2),
    Dropout(0.2),
    Conv2D(32, kernel_size=2, activation='relu'),
    MaxPooling2D(pool_size=2),
    Dropout(0.2),
    Conv2D(64, kernel_size=2, activation='relu'),
    MaxPooling2D(pool_size=2),
    Dropout(0.2),
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    Dense(8, activation='softmax')  # Assuming 8 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    ModelCheckpoint(filepath='best_model.keras', save_best_only=True, monitor='val_accuracy', verbose=1)
]

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.1, callbacks=callbacks)

# Save the model
model.save('respiratory_disease_detectioncnn.keras')

# Plot training and validation accuracy and loss
pyplot.figure()
pyplot.plot(history.history['accuracy'], label='accuracy')
pyplot.plot(history.history['val_accuracy'], label='val_accuracy')
pyplot.legend()
pyplot.title('Accuracy')

pyplot.figure()
pyplot.plot(history.history['loss'], label='loss')
pyplot.plot(history.history['val_loss'], label='val_loss')
pyplot.legend()
pyplot.title('Loss')

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Predict on the test data
predictions = model.predict(X_test)
predicted_classes = numpy.argmax(predictions, axis=1)
true_classes = numpy.argmax(y_test, axis=1)


# ROC Curve
n_classes = 8
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
fig, ax = pyplot.subplots(figsize=(10, 8))
for i in range(n_classes):
    ax.plot(fpr[i], tpr[i], label=f'ROC curve for {encoder.classes_[i]} (area = {roc_auc[i]:.2f})')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='best')
pyplot.show()


from sklearn.metrics import classification_report, confusion_matrix
import numpy as numpy

# Find the intersection of the true and predicted classes
unique_predicted_classes = numpy.unique(predicted_classes)
true_class_labels = numpy.unique(true_classes)

# Only focus on the classes that were both present in true and predicted
common_classes = numpy.intersect1d(true_class_labels, unique_predicted_classes)

# Filter the true and predicted classes to only those common classes
filtered_true_indices = [i for i, c in enumerate(true_classes) if c in common_classes]
filtered_pred_indices = [i for i, c in enumerate(predicted_classes) if c in common_classes]

# Make sure both filtered arrays have the same length
min_length = min(len(filtered_true_indices), len(filtered_pred_indices))
filtered_true_indices = filtered_true_indices[:min_length]
filtered_pred_indices = filtered_pred_indices[:min_length]

true_classes_filtered = true_classes[filtered_true_indices]
predicted_classes_filtered = predicted_classes[filtered_pred_indices]

# Get the target names based on the common classes
adjusted_target_names = encoder.classes_[common_classes]

# Generate classification report and confusion matrix
print("Classification Report:")
print(classification_report(true_classes_filtered, predicted_classes_filtered, target_names=adjusted_target_names))

# Generate confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(true_classes_filtered, predicted_classes_filtered))


# Initialize the SVM model
svm = SVC(kernel='linear', probability=True)
# Assuming X_train and X_test have a shape like (n_samples, height, width, channels)
# Flatten the data to 2D
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Now fit the model with flattened data
svm.fit(X_train_flattened, numpy.argmax(y_train, axis=1))

# Predict and evaluate
y_pred_svm = svm.predict(X_test_flattened)
print(f'SVM Accuracy: {accuracy_score(numpy.argmax(y_test, axis=1), y_pred_svm)}')


# Save the SVM model to a file
joblib.dump(svm, 'svm_respiratory_model_final.keras')


# Predict and evaluate SVM model
y_pred_svm = svm.predict(X_test_flattened)
print(f'SVM Accuracy: {accuracy_score(numpy.argmax(y_test, axis=1), y_pred_svm)}')

# True classes
true_classes = numpy.argmax(y_test, axis=1)

# Find unique classes in the true and predicted data
unique_classes_svm = numpy.unique(numpy.concatenate((true_classes, y_pred_svm)))

# Classification report for SVM with the correct labels
print("SVM Classification Report:")
print(classification_report(true_classes, y_pred_svm, labels=unique_classes_svm, target_names=[encoder.classes_[i] for i in unique_classes_svm]))

# Confusion matrix for SVM
print("SVM Confusion Matrix:")
print(confusion_matrix(true_classes, y_pred_svm, labels=unique_classes_svm))


# Reshape the input data to be 2D (samples, features)
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Fit the XGBoost model with reshaped data
xgboost_model = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgboost_model.fit(X_train_flattened, numpy.argmax(y_train, axis=1))

# Make predictions
y_pred_xgb = xgboost_model.predict(X_test_flattened)

# Evaluate the model
print(f'XGBoost Accuracy: {accuracy_score(numpy.argmax(y_test, axis=1), y_pred_xgb)}')


# Save models
xgboost_model.save_model('xgboost_respiratory_model_final.keras')


# Define the parameter grid for SVM
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Initialize the SVM model
svm = SVC()

# Set up GridSearchCV for SVM
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy', verbose=2)

# Fit the grid search
grid_search_svm.fit(X_train_flattened, numpy.argmax(y_train, axis=1))

# Get the best parameters and score
print("Best parameters for SVM:", grid_search_svm.best_params_)
print("Best cross-validation score for SVM:", grid_search_svm.best_score_)

# Evaluate the tuned SVM model
y_pred_svm_tuned = grid_search_svm.predict(X_test_flattened)
print("Tuned SVM Classification Report:")
print(classification_report(numpy.argmax(y_test, axis=1), y_pred_svm_tuned))


import numpy as numpy
import pandas as pandas
import matplotlib.pyplot as pyplot
from sklearn.metrics import classification_report

# Get classification report for untuned SVM
report_untuned = classification_report(numpy.argmax(y_test, axis=1), y_pred_svm, output_dict=True)

# Get classification report for tuned SVM
report_tuned = classification_report(numpy.argmax(y_test, axis=1), y_pred_svm_tuned, output_dict=True)

# Convert the reports to DataFrames for easier plotting
df_untuned = pandas.DataFrame(report_untuned).transpose()
df_tuned = pandas.DataFrame(report_tuned).transpose()

# Plot comparison for precision, recall, and f1-score
metrics = ['precision', 'recall', 'f1-score']
fig, axs = pyplot.subplots(1, 3, figsize=(18, 6))

bar_width = 0.35  # Set consistent bar width

for i, metric in enumerate(metrics):
    # Indices for positioning
    index = numpy.arange(len(df_untuned.index[:-3]))

    # Plot bars for untuned and tuned SVM
    bars_untuned = axs[i].bar(index, df_untuned[metric][:-3], bar_width, label='Untuned SVM', color='green')
    bars_tuned = axs[i].bar(index + bar_width, df_tuned[metric][:-3], bar_width, label='Tuned SVM', color='purple')

    # Add gridlines for readability
    axs[i].grid(True, linestyle='--', alpha=0.6)

    # Add data labels on top of the bars
    for bar in bars_untuned:
        yval = bar.get_height()
        axs[i].text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 2), ha='center', va='bottom')

    for bar in bars_tuned:
        yval = bar.get_height()
        axs[i].text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 2), ha='center', va='bottom')

    # Set title and labels
    axs[i].set_title(f'{metric.capitalize()} Comparison', fontsize=14)
    axs[i].set_xlabel('Classes', fontsize=12)
    axs[i].set_ylabel(metric.capitalize(), fontsize=12)

    # Set x-ticks and labels
    axs[i].set_xticks(index + bar_width / 2)
    axs[i].set_xticklabels(df_untuned.index[:-3], rotation=45, ha='right', fontsize=11)

    # Add a legend
    axs[i].legend()

# Adjust layout for a better fit
pyplot.tight_layout()
pyplot.show()


# Create an ensemble of classifiers
ensemble = VotingClassifier(estimators=[
    ('svm', svm),
    ('xgboost', xgboost_model)
], voting='hard')

# Fit the ensemble model with the training data
ensemble.fit(X_train_flattened, numpy.argmax(y_train, axis=1))

# Now use the ensemble model to predict
y_pred_ensemble = ensemble.predict(X_test_flattened)


# True classes
true_classes = numpy.argmax(y_test, axis=1)

# For ensemble predictions, ensure to fit or load the ensemble model
# Assuming you have a trained ensemble model
# ensemble.fit(X_train_flattened, numpy.argmax(y_train, axis=1))  # If necessary
y_pred_ensemble = ensemble.predict(X_test_flattened)

# Identify the classes that are present in both true and predicted labels
common_classes = numpy.intersect1d(numpy.unique(true_classes), numpy.unique(y_pred_ensemble))

# Filter out classes that are not in both true and predicted
filtered_true_indices = [i for i, c in enumerate(true_classes) if c in common_classes]
filtered_pred_indices = [i for i, c in enumerate(y_pred_ensemble) if c in common_classes]

# Ensure equal length by taking the minimum size
min_length = min(len(filtered_true_indices), len(filtered_pred_indices))
filtered_true_indices = filtered_true_indices[:min_length]
filtered_pred_indices = filtered_pred_indices[:min_length]

# Get filtered true and predicted classes
filtered_true_classes = true_classes[filtered_true_indices]
filtered_pred_classes = y_pred_ensemble[filtered_pred_indices]

# Get the filtered target names
filtered_target_names = [encoder.classes_[i] for i in common_classes]



# Predict and evaluate XGBoost model
y_pred_xgb = xgboost_model.predict(X_test_flattened)
print(f'XGBoost Accuracy: {accuracy_score(numpy.argmax(y_test, axis=1), y_pred_xgb)}')

# Find unique classes in the true and predicted data
unique_classes_xgb = numpy.unique(numpy.concatenate((true_classes, y_pred_xgb)))

# Classification report for XGBoost with the correct labels
print("XGBoost Classification Report:")
print(classification_report(true_classes, y_pred_xgb, labels=unique_classes_xgb, target_names=[encoder.classes_[i] for i in unique_classes_xgb]))

# Confusion matrix for XGBoost
print("XGBoost Confusion Matrix:")
print(confusion_matrix(true_classes, y_pred_xgb, labels=unique_classes_xgb))

