import pytesseract
import cv2

#img = cv2.imread('Test100.jpg')
# img = cv2.resize(img, None, fx = 0.5, fy = 0.3)
img = cv2.resize(img, (500, 500))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)

text = pytesseract.image_to_string(adaptive_threshold)
print(text)
cv2.imshow('AT', adaptive_threshold)
cv2.waitKey(0)
