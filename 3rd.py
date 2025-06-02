import numpy as np
import cv2
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(0)
result, image = cam.read()
if result:
    cv2.imshow("captured picture", image)
    cv2.waitKey()
    cv2.imwrite("image3.jpg", image)
else:
    print("no image found")

image = cv2.imread("C:/Users/keert/OneDrive/Desktop/ml lab/image2.jpg")
new = cv2.resize(image, (1200, 800))
cv2.imshow('old image', image)
cv2.waitKey()
cv2.imshow('new resized image', new)
cv2.waitKey()
cv2.imwrite("newimage.jpg", new)

blurimage = cv2.blur(image, (50, 50))
cv2.imshow('blurred image', blurimage)
cv2.waitKey()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Lion', gray_image)
cv2.imwrite("gray_image.jpg", gray_image)
cv2.waitKey()

print(image.shape)
h, w = image.shape[:2]
center = (w / 2, h / 2)
print(type(center))
mat = cv2.getRotationMatrix2D(center, 90, 1)
rotating = cv2.warpAffine(image, mat, (h, w))
cv2.imshow('rotated', rotating)
cv2.waitKey()

img_blur = cv2.GaussianBlur(image, (3, 3), sigmaX=0, sigmaY=0)
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey()

src = cv2.imread("C:/Users/keert/OneDrive/Desktop/ml lab/image2.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Gray scale image", src)

th, dst = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY)
cv2.imshow('grey scale image', dst)
cv2.waitKey()

cap = cv2.VideoCapture("C:/Users/keert/OneDrive/Desktop/ml lab/fish1.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    fgmask = fgbg.apply(frame)
    cv2.imshow('fgmask', fgmask)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()

image = cv2.imread("C:/Users/keert/OneDrive/Desktop/ml lab/image2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)

eroded = cv2.erode(gray.copy(), None, iterations=2)
eroded2 = cv2.erode(gray.copy(), None, iterations=5)

cv2.imshow("Eroded 2 times", eroded)
cv2.waitKey()
cv2.imshow("Eroded 5 times", eroded2)
cv2.waitKey(0)
