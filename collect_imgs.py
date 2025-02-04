import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

no_of_classes = 5
dataset_size = 50

cap = cv2.VideoCapture(0)  # define the port
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

for j in range(no_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    print('Collecting data for class {}'.format(j))

    print('Ready? Press "Q" to start collecting data for class {}.'.format(j))
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print('Stopping data collection for class {}'.format(j))
            break

cap.release()
cv2.destroyAllWindows()
