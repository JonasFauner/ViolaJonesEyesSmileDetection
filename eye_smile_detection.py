import cv2
import os

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

image_dir = '159.people/'
# image_dir = 'temp/'

for image_name in os.listdir(image_dir):
    img_orig = cv2.imread(f'{image_dir}{image_name}')
    
    height, width = img_orig.shape[:2]
    new_height = 900
    ratio = new_height / height
    new_width = int(width * ratio)
    img = cv2.resize(img_orig, (new_width, new_height), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(0, 0))

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) for the face in grayscale and color images
        roiGray = gray[y:y+h, x:x+w] 
        roiImg = img[y:y+h, x:x+w]

        # Draw a rectangle around the detected face in the color image
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # eyes = eye_cascade.detectMultiScale(roiGray, scaleFactor=1.05, minNeighbors=6)
        eyes = eye_cascade.detectMultiScale(roiGray, scaleFactor=1.1, minNeighbors=3) #quite good
        smiles = smile_cascade.detectMultiScale(roiGray, scaleFactor=1.1, minNeighbors=30)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roiImg, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roiImg, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
    

    # scale_percent = 300  # Example: scale by 150%
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    # dim = (width, height)

    # # resize image
    # resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    # cv2.imshow('Image', resized_img)
            
    

    cv2.imshow('Image', img)

    key = cv2.waitKey(0) & 0xFF  # Using 0xFF to ensure compatibility across different systems

    # Check if the 'q' key was pressed
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
    # If any other key was pressed, continue to the next imagess
    else:
        cv2.destroyAllWindows()
        continue

    # cv2.waitKey(0)
    



#good
#  eyes = eye_cascade.detectMultiScale(roiGray, scaleFactor=1.07, minNeighbors=3) #quite good
        # smiles = smile_cascade.detectMultiScale(roiGray, scaleFactor=1.05, minNeighbors=30)