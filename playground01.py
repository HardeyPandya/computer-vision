import cv2
import matplotlib.pyplot as plt
import numpy as np

#Abrindo Imagem
img = plt.imread('steel.jpg')
print(img.shape)

#Conversão de uma imagem para outro sistema de cores
#img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def equalizeHistColor(img):
    img_eq = np.zeros(img.shape)
    
    for i in range(img.shape[2]):
        img_eq[:,:,i] = cv2.equalizeHist(img[:,:,i])
    
    return img_eq

img_eq = equalizeHistColor(img)

plt.figure(figsize=(5,5)); plt.title("img"); fig = plt.imshow(img, 'gray')
plt.figure(figsize=(5,5)); plt.title("img_eq"); fig = plt.imshow(img_eq, 'gray')
