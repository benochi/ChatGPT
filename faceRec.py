# import necessary libraries
import dlib
import cv2

# load the trained model
model = dlib.simple_object_detector("detector.svm")

# open the camera
cap = cv2.VideoCapture(0)

while True:
    # read a frame from the camera
    _, frame = cap.read()

    # detect faces in the frame
    faces = model(frame)

    # loop over the detected faces
    for face in faces:
        # get the bounding box for the face
        x, y, w, h = face.left(), face.top(), face.right(), face.bottom()

        # draw a box around the face
        cv2.rectangle(frame, (x,y), (w,h), (0,255,0), 2)

    # show the frame
    cv2.imshow("Frame", frame)

    # check for user input
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
