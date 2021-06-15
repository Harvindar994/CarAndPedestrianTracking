import cv2

videoFile = 'video.mp4'

# pre-trained car classifier.
classifierFile = 'cars.xml'
pedestrianFile = "haarcascade_fullbody.xml"

# reading Video file using open cv2
video = cv2.VideoCapture(videoFile)

# creating classifier using cv2.
CarClassifier = cv2.CascadeClassifier(classifierFile)
PedestrianClassifier = cv2.CascadeClassifier(pedestrianFile)

# here we will read video file frame by frame.
while True:
    readStatus, frame = video.read()

    # if frame read successfully.
    if readStatus:
        # converting image into gray scale.
        GrayScaleImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detecting cars in the given image.
        CarsCoord = CarClassifier.detectMultiScale(GrayScaleImage)

        # detecting coord of pedestrians.
        pedestrianCoord = PedestrianClassifier.detectMultiScale(GrayScaleImage)

        # drawing rectangles on the color image using Cars Coord that we created using Classifier.
        for rect in CarsCoord:
            x, y, width, height = rect
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 1)

        for rect in pedestrianCoord:
            x, y, width, height = rect
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 255), 1)

        # displaying the image that we loaded.
        cv2.imshow("Cars Image", frame)
        cv2.waitKey(1)


