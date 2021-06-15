import cv2
import numpy as np
import matplotlib.pyplot as plt

def graph(input):
    plt.clf()
    plt.plot(input)
    plt.title("Persons detected over time")
    plt.xlabel("frame")
    plt.grid()
    plt.savefig("plot.png", transparent=True)

def add_plot(path, frame):
    plot_image = cv2.imread(path)
    dim = (int(frame.shape[0] * 0.4), int(frame.shape[0] * 0.4))
    plot_image = cv2.resize(plot_image, dim, interpolation = cv2.INTER_AREA)
    height_image, width_image, channels_images = plot_image.shape
    x, y = 40, 0
    for i in range(height_image):
        for j in range(width_image):
            frame[i + x][j + y] = plot_image[i][j]
    return frame

def object_detection(frame):
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (320, 320), [0, 0, 0], 1, crop=False)  # reduce to 320
    net.setInput(blob)
    outs = net.forward(outputlayers)
    return blob, outs

#Load YOLO and COCO classes
net = cv2.dnn.readNet('yolov3-320.weights', 'yolov3-320.cfg')
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layers_names = net.getLayerNames()
outputlayers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# Loading image
cap = cv2.VideoCapture("london.mov")
_, frame = cap.read()
height, width, channels = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output_video = cv2.VideoWriter('outpy.avi', fourcc, 24, (720, 480))

#OpenCV and detection parameters
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3)) #random color for rectangles
confThreshold = 0.4
NMSThreshold = 0.3 #overlapping boxes (0.3 is agressive)
historic_detection = []


while True:
    #empty list creation
    class_ids = []
    confidences = []
    boxes = []

    # read contend
    _, frame = cap.read()

    #detecting object with blob
    blob, outs = object_detection(frame)
    
    #find detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confThreshold:
                # box shape
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # box coordinate
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, NMSThreshold)

    #draw detection
    person_detected = 0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if classes[class_ids[i]] == "person":
                person_detected += 1
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence * 100, 0)) + "%", (x, y + 30), font, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Person(s) detected = " + str(person_detected), (10, 30), font, 2, (255, 255, 255, 15))

    #plot historic
    historic_detection.append(person_detected)
    graph(historic_detection)
    frame = add_plot('plot.png', frame)
    cv2.imshow("Frame", frame)

    #save video frame
    frame_output = cv2.resize(frame, (720, 480), interpolation=cv2.INTER_AREA)
    output_video.write(frame_output)

    #stop loop
    key = cv2.waitKey
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
output_video.release()
cv2.destroyAllWindows()