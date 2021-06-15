import cv2

imageFile = 'image.jpg'

# pre-trained car classifier.
classifierFile = 'cars.xml'

# reading image file using open cv2
image = cv2.imread(imageFile)

# creating classifier using cv2.
CarClassifier = cv2.CascadeClassifier(classifierFile)

# converting image into gray scale.
GrayScaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detecting cars in the given image.
CarsCoord = CarClassifier.detectMultiScale(GrayScaleImage)

# drawing rectangles on the color image using Cars Coord that we created using Classifier.
for rect in CarsCoord:
    x, y, width, height = rect
    cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 1)

# displaying the image that we loaded.
cv2.imshow("Cars Image", image)
cv2.waitKey()


