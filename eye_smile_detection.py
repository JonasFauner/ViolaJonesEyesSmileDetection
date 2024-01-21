# Importing the OpenCV library
import cv2
import os

# Loading Haar cascade classifiers for face, eyes, and smiles detection
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Directory containing the images
image_dir = '159.people/'

# image_i = 0
# show_images = [7,8,9,24,26]


# Iterating through each image in the directory
for image_name in os.listdir(image_dir):
    #count images shown
    # image_i += 1 
    # if image_i not in show_images:
    #     continue

    # Reading the image
    img_orig = cv2.imread(f'{image_dir}{image_name}')
    
    # Getting the height and width of the original image
    height, width = img_orig.shape[:2]
    # Setting a new height and calculating the corresponding width to maintain aspect ratio
    new_height = 600
    ratio = new_height / height
    new_width = int(width * ratio)
    # Resizing the image
    img = cv2.resize(img_orig, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Converting the resized image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detecting faces in the grayscale image
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2 , minNeighbors=5, minSize=(30, 30))

    # Iterating over each detected face
    for (x, y, w, h) in faces:
        # Extracting the region of interest (ROI) for the face in grayscale and color images
        roiGray = gray[y:y+h, x:x+w] 
        roiImg = img[y:y+h, x:x+w]

        # Drawing a rectangle around the detected face in the color image
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Detecting eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roiGray, scaleFactor=1.1, minNeighbors=7, minSize=(20, 20)) 
        # Detecting smiles within the face ROI
        smiles = smile_cascade.detectMultiScale(roiGray, scaleFactor=1.3 , minNeighbors=20 , minSize=(25, 25))

        # Drawing rectangles around detected eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roiImg, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

        # Drawing rectangles around detected smiles
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roiImg, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
    
    print("Displaying image:  ", image_name)
    # Displaying the image with detected features
    cv2.imshow('Image', img)

    # Waiting for a key press
    key = cv2.waitKey(0) & 0xFF  # Using 0xFF to ensure compatibility across different systems

    # Check if the 'q' key was pressed to exit
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
    # If any other key was pressed, continue to the next image
    else:
        cv2.destroyAllWindows()
        continue
