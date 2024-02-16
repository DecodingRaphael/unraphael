import cv2
import sys
from matplotlib import pyplot as plt

imagePath = sys.argv[1]

image = cv2.imread("../../data/interim/no_bg/output_8_London_OrderStJohn.jpg")

# Display the image
plt.imshow(image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
bodyCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(40, 40)
)

bodies = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
)


print("[INFO] Found {0} Faces!".format(len(faces)))

for (x, y, w, h) in faces:
    
    # Calculate the center of the bounding box
    center_x = x + w // 2
    center_y = y + h // 2
        
    # Calculate the new width and height
    w = int(w * 2.50)
    h = int(h * 1.75)

    # Adjust the coordinates to make the bounding box twice as big        
    x = max(0, center_x - w // 2)
    y = max(0, center_y - h // 2)
    
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    roi_color = image[y:y + h, x:x + w]
    
    print("[INFO] Object found. Saving locally.")
    cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)

status = cv2.imwrite('faces_detected.jpg', image)
print("[INFO] Image faces_detected.jpg written to filesystem: ", status)