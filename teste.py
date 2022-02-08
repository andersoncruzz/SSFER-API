import cv2
from ssfer import SSFER

img = cv2.imread("input/teste.png")

ssfer = SSFER()
probs = ssfer.predict(img)
print(probs)