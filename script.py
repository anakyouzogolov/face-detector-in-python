import cv2

# loading image for testing
my_img = cv2.imread("img.jpg")

# change our image to grayscale image 
gray_img = cv2.cvtColor(my_img, cv2.COLOR_BGR2BGRA)


# initialize the face recognizer (default face haar cascade)
face_cascade = cv2.CascadeClassifier("C:\\Users\\zogopy\\AppData\\Local\\Programs\Python\\Python38-32\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")


# detect all the faces in our image
faces = face_cascade.detectMultiScale(gray_img)

# print the number of faces in our image
print("faces detected are :", len(faces))


# for every face draw a rectangle with blue color
for x, y, width, height in faces:
    cv2.rectangle(gray_img, (x, y), (x + width, y+ height), color=(255, 0, 0), thickness=3)


# save the image
cv2.imwrite("faces_detected.jpg", gray_img)