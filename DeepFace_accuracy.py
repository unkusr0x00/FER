import os
import pandas as pd
from deepface import DeepFace
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

base_dir = "Datasets/RAF-FER-SFEW-AN"

# Emotion classes
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

images_per_category = {
    'happy': 17299,
    'neutral': 12928,
    'sad': 11161,
    'angry': 9415,
    'surprise': 9302,
    'fear': 8649,
    'disgust': 4767
}

# Store results
results = []
true_labels = []
pred_labels = []

for emotion in emotions:
    emotion_dir = os.path.join(base_dir, emotion)
    all_images = os.listdir(emotion_dir)
    # Wrap the image processing loop with tqdm for a progress bar
    for image in tqdm(all_images, desc=f"Processing {emotion} images"):
        img_path = os.path.join(emotion_dir, image)
        try:
            analysis = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)
            predicted_emotion = analysis[0]
            predicted = predicted_emotion['dominant_emotion']
            # Append true label and predicted label
            true_labels.append(emotion)
            pred_labels.append(predicted)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Calculate overall accuracy
overall_accuracy = accuracy_score(true_labels, pred_labels) * 100

# Calculate precision, recall, and F1-score for each class
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, labels=emotions, average=None)

# Calculate metrics for each class and store in a DataFrame
metrics_data = {
    'Emotion': emotions,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
}

df_metrics = pd.DataFrame(metrics_data)

# Display the accuracy and metrics table
print(f"Overall Accuracy: {overall_accuracy}%")
print(df_metrics)

# Convert emotions to a list for indexing
emotions = df_metrics['Emotion'].tolist()

# Calculate weights (number of images per category)
weights = [images_per_category[emotion] for emotion in emotions]

# Calculate weighted metrics
weighted_precision = sum(df_metrics['Precision'] * weights) / sum(weights)
weighted_recall = sum(df_metrics['Recall'] * weights) / sum(weights)
weighted_f1 = sum(df_metrics['F1-Score'] * weights) / sum(weights)

print(f"Weighted Precision: {weighted_precision}")
print(f"Weighted Recall: {weighted_recall}")
print(f"Weighted F1-Score: {weighted_f1}")