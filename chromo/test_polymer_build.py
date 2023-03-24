import cv2
import imutils
import numpy as np
import os


def test_image(new_image_name, control_image_name):
    cwd = os.getcwd()  # get pathname of current working directory
    # Load the two images
    new_location1 = cwd + "/chromo/" + new_image_name + ".png"
    old_location1 = cwd + "/chromo/" + control_image_name + ".png"

    new_image = cv2.imread(new_location1)
    control_image = cv2.imread(old_location1)
    # Resize images if necessary
    new_image = cv2.resize(new_image, (600, 360))
    control_image = cv2.resize(control_image, (600, 360))

    img_height = new_image.shape[0]

    # Grayscale
    gray1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(control_image, cv2.COLOR_BGR2GRAY)

    # Calculate absolute difference between two arrays
    diff = cv2.absdiff(gray1, gray2)
    cv2.imshow("diff(new_image, control_image)", diff)

    # Apply threshold. Apply both THRESH_BINARY and THRESH_OTSU
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Threshold", thresh)

    #    Dilation
    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    cv2.imshow("Dilate", dilate)

    # Calculate contours
    contours = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for contour in contours:
        if cv2.contourArea(contour) > 100:
            # Calculate bounding box around contour
            x, y, w, h = cv2.boundingRect(contour)
            # Draw rectangle - bounding box on both images
            cv2.rectangle(new_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(control_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Show images with rectangles on differences
    x = np.zeros((img_height, 10, 3), np.uint8)
    result = np.hstack((new_image, x, control_image))
    cv2.imshow("Differences", result)

    user_opinion = input("Does your output approximately match the reference image.  \n Note: the polymer may be in a different orientation ? Enter yes or no: ")

    if user_opinion == "yes":
        print("Test passed")
    if user_opinion == "no":
        print("Test not passed")

    quit()
