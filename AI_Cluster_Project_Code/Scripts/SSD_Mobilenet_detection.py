# SSD_MOBILENET v3 OBJECT DETECTION FROM PICTURES & VIDEO - FINAL PROJECT


import cv2
import numpy as np
import random
import time


# -----------------------------------------------------------------------------------------------------------
# SECTION 1 IMAGE DETECTION
# UNCOMMENT FOR IMAGES

# Reading image with OpenCV library
# We can change the image name depending on which image we which to work with
Original_image=cv2.imread('images/Example_1.png')
# # Showing image shape
print('Image shape:', Original_image.shape)
# Getting only height and width of image
h, w = Original_image.shape[:2]
# Getting blob from input image
blob = cv2.dnn.blobFromImage(Original_image, 1 / 127.5, (320, 320),swapRB=True, crop=False)
print('Blob shape:', blob.shape)
# Load COCO class labels from coco.names file

# Load COCO class labels from coco.names file
with open('coco.names') as f:
    # Getting labels reading every line
    # and putting them into the list
    Labels = [line.strip() for line in f]
print('Labels:')
print(Labels)


bboxes = []
confidences = []
class_numbers = []
# # Setting minimum confidence to eliminate weak predictions
confidence_minimum = 0.4


# import configuration file
configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# import weights file
weightsPath='frozen_inference_graph.pb'

# Load pre-trained YOLO v4 Objects Detector
ssd_network = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')

# Get list with names of all layers from network
layers_names_all = ssd_network.getLayerNames()
print(layers_names_all)
print(len(layers_names_all))


ssd_network.setInput(blob) # setting blob as input to the network
# Implementing  pass with our blob through network
output_from_ssd_network = ssd_network.forward()

# Generating colours for representing every detected object
colours = np.random.uniform(0, 255, size=(len(Labels), 3))

counter = 1
# For loop for detections and output
for i in np.arange(0,  output_from_ssd_network.shape[2]):
# for i in np.arange(0, output_from_ssd_network.shape[2]):
	# Get value of probability class
    confidence_current =  output_from_ssd_network[0, 0, i, 2]
    # confidence_current = output_from_ssd_network[0, 0, i, 2]

    # Eliminate weak predictions with minimum probability
    if confidence_current > confidence_minimum:
        #Compute current class
        current_class = int( output_from_ssd_network[0, 0, i, 1]-1)
        # Configure bounding box coordinates to the initial image size
        bbox_current =  output_from_ssd_network[0, 0, i, 3:7] * np.array([w, h, w, h])
        (top_x, top_y, bottom_x, bottom_y) = bbox_current.astype("int")
        confidences.append(float(confidence_current))
        class_numbers.append(current_class)
        # Showing labels of the detected objects
        print('Object {0}: {1}'.format(counter, Labels[int(class_numbers[i])]))
        counter += 1
        colour_box_current = colours[class_numbers[i]].tolist()
        # Drawing bounding box on the original image
        cv2.rectangle(Original_image, (top_x, top_y),
                      (bottom_x,bottom_y),
                      colour_box_current, 2)
        # Preparing text with label and confidence for current bounding box
        text_box_current = '{}: {:.4f}'.format(Labels[int(class_numbers[i])],
                                               confidences[i])
        cv2.putText(Original_image, text_box_current, (top_x, top_y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)

# show the output image
cv2.imshow("Detections", Original_image)
cv2.waitKey(0)
# -------------------------------------END OF SECTION 1-----------------------------------------------------------------







# # SECTION 2 ---------------------------------------------------------------------------------------------------------
# # UNCOMMENT FOR VIDEO DETECTION
video = cv2.VideoCapture('videos_ssd/video4.avi')
    # Define recorder variable for writing frames
recorder = None
    # Variables for height and width
h, w = None, None



# Load COCO class labels from coco.names file
with open('coco.names') as f:
    # Getting labels reading every line
    # and putting them into the list
    Labels = [line.strip() for line in f]
print('Labels:')
print(Labels)


# import configuration file
configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# import weights file
weightsPath='frozen_inference_graph.pb'

# Load pre-trained YOLO v4 Objects Detector
ssd_network = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')

# Get list with names of all layers from network
layers_names_all = ssd_network.getLayerNames()
print(layers_names_all)
print(len(layers_names_all))

# Setting minimum confidence to eliminate weak predictions
confidence_minimum = 0.6
# Setting threshold for filtering weak bounding boxes
nms_threshold = 0.3

# Generating colours for representing every detected object
colours = np.random.randint(0, 255, size=(len(Labels), 3), dtype='uint8')
# Generating colours for representing every detected object
# colours = np.random.uniform(0, 255, size=(len(Labels), 3))


# Define variable for counting frames
f = 0

# Define variable for time
t = 0


while True:
    # Capture frame-by-frame
    ret, frame = video.read()
    if not ret:
        break
    # Get spatial dimensions of frame
    if w is None or h is None:
        h, w = frame.shape[:2]

    # Getting blob from input image
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

    # Implementing forward pass with our blob through output layers
    ssd_network.setInput(blob)  # setting blob as input to the network
    start = time.time()
    output_from_ssd_network = ssd_network.forward()
    end = time.time()

    # Increase counters for frames and total time
    f += 1
    t += end - start

    print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))

    bboxes = []
    confidences = []
    class_numbers = []

    counter = 1
    # For loop for detections and output
    for i in np.arange(0,  output_from_ssd_network.shape[2]):
    # for i in np.arange(0, output_from_ssd_network.shape[2]):
    	# Get value of probability class

        confidence_current =  output_from_ssd_network[0, 0, i, 2]
        # confidence_current = output_from_ssd_network[0, 0, i, 2]

        # Eliminate weak predictions with minimum probability
        if confidence_current > confidence_minimum:
            #Compute current class
            current_class = int( output_from_ssd_network[0, 0, i, 1]-1)
            # Configure bounding box coordinates to the initial image size
            bbox_current =  output_from_ssd_network[0, 0, i, 3:7] * np.array([w, h, w, h])
            (top_x, top_y, bottom_x, bottom_y) = bbox_current.astype("int")
            confidences.append(float(confidence_current))
            class_numbers.append(current_class)
            # Showing labels of the detected objects
            print('Object {0}: {1}'.format(counter, Labels[int(class_numbers[i])]))
            counter += 1
            colour_box_current = colours[class_numbers[i]].tolist()
            # Drawing bounding box on the original image
            cv2.rectangle(frame,(top_x, top_y),(bottom_x,bottom_y),colour_box_current, 2)
            # Preparing text with label and confidence for current bounding box
            text_box_current = '{}: {:.4f}'.format(Labels[int(class_numbers[i])],
                                                   confidences[i])
            cv2.putText(frame, text_box_current, (top_x, top_y - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)

    # Record in new video output
    if recorder is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        recorder = cv2.VideoWriter('videos_ssd/result-video4.mp4', fourcc, 30,
                                   (frame.shape[1], frame.shape[0]), True)
    recorder.write(frame)

# Printing final results
print('FPS:', round((f / t), 1))

# Releasing video reader and writer
video.release()
recorder.release()
# -----------------------------  END OF SECTION 2 ------------------------------------------------------------