import cv2

img=cv2.imread("cai.jpg")
cv2.imshow('ss', img)

myroi = img[0:100, 0:50]
cv2.imshow('tt', myroi)
cv2.waitKey(0)