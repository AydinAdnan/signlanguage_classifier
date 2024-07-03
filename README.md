**How It Works**


**Collecting Images**

-The collect_imgs.py script is used to capture images for different gesture classes using a webcam. The script saves these images in a specified directory, organized by class.

    DATA_DIR: Directory where images will be stored.
    no_of_classes: Number of different gesture classes.
    dataset_size: Number of images per class.
    
**Creating Dataset**

-The create_dataset.py script processes the collected images to extract hand landmarks using MediaPipe. The coordinates of these landmarks are then saved in a pickle file (data.pickle).*If you want to know how Mediapipe works for hands run the detect_hands_allpts.py*

**Training Classifier**
*The train_classifier.py script loads the landmark data and trains a Random Forest classifier. The trained model is saved in a pickle file (model.p).


**Testing**
-The test.py script uses the trained model to recognize hand gestures in real-time from a live webcam feed. The script captures video, detects hand landmarks, predicts the gesture, and displays the result on the video feed.
