import cv2
import matplotlib.pyplot as plt
import numpy as np

#Abrindo Imagem
img = plt.imread('steel.jpg')
print(img.shape)

#Conversão de uma imagem para outro sistema de cores
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Visualização de Imagem individual
plt.figure(figsize=(5,5)); plt.title("img_gray"); fig = plt.imshow(img_gray, 'gray')


print(img_gray[:10,:10])
img_gray = img_gray[:10, :10]

def integral(img_gray):
    x, y = img_gray.shape
    img_integral = np.zeros((y,x), dtype=np.uint64)
    
    for iy in range(0,y):
        for ix in range(0,x):
            if   iy==0 and ix==0:
                img_integral[0,0] = img_gray[0,0]
            elif iy==0:
                img_integral[0,ix] = img_integral[0,ix-1] + img_gray[0,ix]
            elif ix==0:
                img_integral[iy,0] = img_integral[iy-1,0] + img_gray[iy,0]
            else:
                img_integral[iy,ix] = img_gray[iy,ix] - \
                    img_integral[iy-1,ix-1] + \
                    img_integral[iy  ,ix-1] + \
                    img_integral[iy-1,ix  ]
    
    return img_integral

def sum_integral(img_integral, start, end):
    sy, sx = start
    ey, ex = end
    
    result  = img_integral[sy-1,sx-1]
    result -= img_integral[sy-1,ex-1]
    result -= img_integral[ey-1,sx-1]
    result += img_integral[ey-1,ex-1]
    
    #result = img_integral[sy-1,sx-1] - \
    #    img_integral[sy-1,ex  ] - \
    #    img_integral[ey  ,sx-1] + \
    #    img_integral[ey  ,ex  ]
    
    return result

img_gray = np.ones((10,10), dtype=np.uint8)

img_integral2 = integral(img_gray)

print(img_gray)
print(img_integral2)

print(img_gray.dtype, img_gray.shape)
print(img_integral2.dtype, img_integral2.shape)

start = (3,3)
end = (6,6)
print(np.sum(img_gray[ start[0]:end[0], start[1]:end[1] ]))
print(sum_integral(img_integral2, start, end))
