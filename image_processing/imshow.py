import cv2

image = cv2.imread("C:/Users/lixiaojun/Downloads/98762b790902ade4f4cabae079044fbd.jpg")
cv2.imshow('img', image)

key = cv2.waitKey(0)
if key == 27:  # 按esc键时，关闭所有窗口
    print(key)
    cv2.destroyAllWindows()
