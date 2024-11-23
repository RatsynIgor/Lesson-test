print('lesson 11. Image treatment by AI')

import cv2

image_path = '3xcats.jpg'

handle_cat_face = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')
image_cat = cv2.imread(image_path)

cat_face_data = handle_cat_face.detectMultiScale(image_cat)
print('cat face', cat_face_data)

colors = [(0, 255, 255), (255, 0, 255), (0, 0, 255)]
index = 0

for(x, y, w, h) in cat_face_data:
    cv2.rectangle(image_cat, (x, y), (x+w, y+h), colors[index], 3)
    index += 1

cv2.imshow('cat', image_cat)
cv2.waitKey()

