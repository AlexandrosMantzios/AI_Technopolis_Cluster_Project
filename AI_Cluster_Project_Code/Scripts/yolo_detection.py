
# YOLO v4 OBJECT DETECTION FROM PICTURES & VIDEO - FINAL PROJECT

# Import libraries
import numpy as np
import cv2
import time


# FUNCTIONS - RETAIN FOR BOTH IMAGE AND VIDEO DETECTION

# Function for model
def model():
    # Load pre-trained YOLO v4 Objects Detector
    yolo_network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov4.cfg', 'yolo-coco-data/yolov4.weights')
    # Get list with names of all layers from YOLO v4 network
    layers_names_all = yolo_network.getLayerNames()
    print(layers_names_all)
    # Getting output layers YOLO v4
    # with function that returns indexes of layers with unconnected outputs
    layers_names_output = [layers_names_all[i[0] - 1] for i in yolo_network.getUnconnectedOutLayers()]
    print()
    print(layers_names_output)
    confidence_minimum = 0.5
    threshold = 0.3
    # Implementing forward pass with our blob through output layers
    yolo_network.setInput(blob)  # setting blob as input to the network
    output_from_yolo_network = yolo_network.forward(layers_names_output)
    # return yolo_network,output_from_yolo_network,confidence_minimum,threshold
    output(confidence_minimum,output_from_yolo_network)
    return confidence_minimum, threshold




# Function for detections
def detections(object,colours):
    counter = 1
    if len(results) > 0:
        for i in results.flatten():
            # Showing labels of the detected objects
            print('Object {0}: {1}'.format(counter, labels[int(class_numbers[i])]))
            counter += 1
            # Bounding box coordinates,
            x_min, y_min = bboxes[i][0], bboxes[i][1]
            box_width, box_height = bboxes[i][2], bboxes[i][3]
            colour_box_current = colours[class_numbers[i]].tolist()
            # Drawing bounding box on the original image
            cv2.rectangle(object, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)
            # Preparing text with label and confidence for current bounding box
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                   confidences[i])
            cv2.putText(object, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)




# Function for going through all output layers after feed forward pass
def output(confidence_minimum,output_from_yolo_network):
    for result in output_from_yolo_network:
        for detected_objects in result:
            probabilities = detected_objects[5:]
            # Indexes of  class with the maximum probability
            current_class = np.argmax(probabilities)
            # Get value of probability class
            confidence_current = probabilities[current_class]
            # Eliminate weak predictions with minimum probability
            if confidence_current > confidence_minimum:
                # Scale bounding box coordinates to the initial image size
                box_current = detected_objects[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))
                bboxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(current_class)


#
# -----------------------------------------------------------------------------------------------------------

# SECTION 1 IMAGE DETECTION
# UNCOMMENT FOR IMAGE DETECTION
# Reading image with OpenCV library
# We can change the image name depending on which image we which to work with
Original_image = cv2.imread('images/Example_1.png')
# # Showing image shape
print('Image shape:', Original_image.shape)
# Getting only height and width of image
h, w = Original_image.shape[:2]
# Getting blob from input image
blob = cv2.dnn.blobFromImage(Original_image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
print('Blob shape:', blob.shape)
# Transpose blob image to make channels come at the end
blob_to_show = blob[0, :, :, :].transpose(1, 2, 0)
print(blob_to_show.shape)
# Load COCO class labels from coco.names file
with open('yolo-coco-data/coco.names') as f:
    # Getting labels reading every line
    # and putting them into the list
    labels = [line.strip() for line in f]
print('Labels:')
print(labels)

bboxes = []
confidences = []
class_numbers = []

# # Setting minimum confidence to eliminate weak predictions
confidence_minimum = 0.5

# # Setting threshold for filtering weak bounding boxes
threshold = 0.4

# Generating colours for representing every detected object
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# Applying model function
model()

# Implementing non-maximum suppression on  bounding boxes
results = cv2.dnn.NMSBoxes(bboxes, confidences,
                           confidence_minimum, threshold)

# Applying function for detections
detections(Original_image,colours)

# Show Original Image with Detected Objects
cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
cv2.imshow('Detections', Original_image)
cv2.waitKey(0)
cv2.destroyWindow('Detections')
#-------------------------------------END OF SECTION 1-----------------------------------------------------------------


# # SECTION 2 ------------------------------------------------------------------------------------------
# # UNCOMMENT FOR VIDEO DETECTION
# For loop for reading 11 selected videos.
for x in range (1,12):
    video = cv2.VideoCapture('videos/video{}.avi'.format(x))
    # Define recorder variable for writing frames
    recorder = None
    # Variables for height and width
    h, w = None, None


    # Load COCO class labels from coco.names file
    with open('yolo-coco-data/coco.names') as f:
        # Getting labels reading every line
        # and putting them into the list
        labels = [line.strip() for line in f]
    print('Labels:')
    print(labels)

    # Load pre-trained YOLO v4 Objects Detector
    yolo_network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov4.cfg',
                                         'yolo-coco-data/yolov4.weights')

    # Getting list with names of all layers from YOLO v3 network
    layers_names_all = yolo_network.getLayerNames()

    # Getting output layers YOLO v4
    # with function that returns indexes of layers with unconnected outputs
    layers_names_output = \
        [layers_names_all[i[0] - 1] for i in yolo_network.getUnconnectedOutLayers()]
    print()
    print(layers_names_output)
    #

    # Setting minimum confidence to eliminate weak predictions
    confidence_minimum = 0.5
    # Setting threshold for filtering weak bounding boxes
    threshold = 0.3

    # Generating colours for representing every detected object
    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Define variable for counting frames
    f = 0

    # Define variable for time
    t = 0

    # Loop for catching frames
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
        yolo_network.setInput(blob)  # setting blob as input to the network
        start = time.time()
        output_from_yolo_network = yolo_network.forward(layers_names_output)
        end = time.time()

        # Increase counters for frames and total time
        f += 1
        t += end - start

        print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))

        bboxes = []
        confidences = []
        class_numbers = []

        # Call output function
        output(confidence_minimum,output_from_yolo_network)
        results = cv2.dnn.NMSBoxes(bboxes, confidences,
                                   confidence_minimum, threshold)

        # Call detections function
        detections(frame,colours)


        # Record in new video output
        if recorder is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            recorder = cv2.VideoWriter('videos/result-video{}.mp4'.format(x), fourcc, 30,
                                     (frame.shape[1], frame.shape[0]), True)
        recorder.write(frame)

    # Printing final results
    print('FPS:', round((f / t), 1))


    # Releasing video reader and writer
    video.release()
    recorder.release()
# -----------------------------  END OF SECTION 2 ------------------------------------------------------------
