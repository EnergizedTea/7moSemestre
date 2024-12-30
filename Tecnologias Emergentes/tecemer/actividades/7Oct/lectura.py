import matplotlib.pyplot as plt
import matplotlib.image as mimg
import cv2
import numpy

image = mimg.imread('lena.tif')

plt.imshow(image)
plt.show()

m, n, c = image.shape
print(f'Ancho:  {n}, Alto:  {m}, Canales:   {c}')
pixel = image[0,0]
print(f'Pixel (0,0): {pixel}')

""" for i in range(len(image)):
    print(image[i]) """

# Esta era la respuesta

""" for i in range(m):
    for j in range(n):
        pixel = image[i,j]
        print(pixel, end=' ')
    print() """

# Histograma
'''plt.hist(image.ravel(), bins=256, range=[0,256])
plt.show()'''

color = ('r','g','b')
# Matplotlib lee las imagenes en rgb
# cv2 lo lee en bgr

plt.subplot(1,2,1)
plt.title('Histograma')
plt.xlabel('Intensidad')
plt.ylabel('Frecuencia')
for i, col in enumerate(color):
    hist = cv2.calcHist([image], [i], None, [256], [0,256])
    plt.plot(hist, color=col)
plt.subplot(1,2,2)
plt.title('Imagen Original')
plt.imshow(image)
plt.show()

newimage = image.copy()

for i in range(m):
    for j in range(n):
        newimage[i,j][0]= newimage[i,j][0]/5
plt.imshow(newimage)
plt.show()

# TODO: TRABAJAR EN EL GITHUB, A LO MEJOR CREAR UNA API QUE 
# RECIBE UNA IMAGEN Y CREA UN HISTOGRAMA CON RESPECTO A ELLO

# AHORA TOCA, MANIPULACION DEL CANAL ROJO.



""" image = cv2.imread('lena.tif')
cv2.imshow('Lena', image) 
cv2.waitKey(0)
cv2.destroyAllWindows() """

# TODO: Convertir todo a funciones y crear funcion del brillo
# Deben ser modificar canal rojo, la binarización de la imagen, el canal de brillo, etc