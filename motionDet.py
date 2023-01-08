import cv2

# start the webcam
cap = cv2.VideoCapture(0)

# take the first frame as the reference frame
_, ref_frame = cap.read()

while True:
    # read a frame from the webcam
    _, frame = cap.read()

    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # compute the absolute difference between the reference frame and the current frame
    diff = cv2.absdiff(ref_frame, gray)

    # threshold the difference image
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # show the thresholded difference image
    cv2.imshow("Thresholded Difference", thresh)

    # check for user input
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        # update the reference frame
        ref_frame = gray

# release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
