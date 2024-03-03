import os

import pandas as pd
from deepface import DeepFace

base_dir = "../Datasets/RAF-FER-SFEW-AN"


# Emotion classes
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Store results
results = {emotion: {'correct': 0, 'total': 0} for emotion in emotions}

for emotion in emotions:
    emotion_dir = os.path.join(base_dir, emotion)
    all_images = os.listdir(emotion_dir)
    for image in all_images:
        img_path = os.path.join(emotion_dir, image)
        try:
            analysis = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)
            predicted_emotion = analysis[0]
            predicted = predicted_emotion['dominant_emotion']
            # Increment correct count if prediction matches the folder name
            if predicted == emotion:
                results[emotion]['correct'] += 1
            results[emotion]['total'] += 1

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Calculate accuracy for each class and store in a DataFrame
accuracy_data = {
    'Emotion': [],
    'Accuracy': []
}

for emotion, result in results.items():
    accuracy = (result['correct'] / result['total']) * 100 if result['total'] > 0 else 0
    accuracy_data['Emotion'].append(emotion)
    accuracy_data['Accuracy'].append(accuracy)

df_accuracy = pd.DataFrame(accuracy_data)
# Display the accuracy table
print(df_accuracy)