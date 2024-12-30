import matplotlib.pyplot as plt
import matplotlib.image as mimg
import numpy as np
import cv2

image = mimg.imread('lena.tif')

class ImageManipulation():
    def __init__(self, image):
        self.image = image

    # Funcion que cambia la imagen con la que se trabaja
    def set_img(self, image):
        self.image = image

    # función que muestra y describe la imagen
    def des_img(self):
        plt.imshow(self.image)
        plt.show()

        m, n, c = self.image.shape
        print(f'Ancho:  {n}, Alto:  {m}, Canales:   {c}')
        pixel = self.image[0,0]
        print(f'Pixel (0,0): {pixel}')

    # Funcion para imprimir en terminal valores 
    def imp_val(self):
        m, n = self.image.shape
        for i in range(m):
            for j in range(n):
                pixel = self.image[i,j]
                print(pixel, end=' ')
            print() 

    # Funcion para mandar crear el histograma 
    def his_img(self):
        color = ('r','g','b')
        plt.subplot(1,2,1)
        plt.title('Histograma')
        plt.xlabel('Intensidad')
        plt.ylabel('Frecuencia')
        for i, col in enumerate(color):
            hist = cv2.calcHist([self.image], [i], None, [256], [0,256])
            plt.plot(hist, color=col)
        plt.subplot(1,2,2)
        plt.title('Imagen Original')
        plt.imshow(self.image)
        plt.show()

    # Funcion para minimizar la existencia de un color especifico
    def rgb_img(self, rgb=int):
        newimage = self.image.copy()
        m, n, c = self.image.shape
        for i in range(m):
            for j in range(n):
                newimage[i,j][rgb]= newimage[i,j][rgb]/10
        plt.imshow(newimage)
        plt.show()
        return newimage
    
    # Funcion para convertir imagen a blanco y negro
    def bnw_img(self):
        bnw_image = np.mean(self.image, axis=2)
        plt.imshow(bnw_image, cmap='gray')
        plt.show()
        return bnw_image
    
    # Funcion para binarizar imagen
    def bin_img(self):
        bnw_image = self.bnw_img()
        bin_image = (bnw_image > 128) * 255
        plt.imshow(bin_image, cmap='gray')
        plt.show()
        return bin_image

    # Funcion para aumentar el canal de brillo
    def bri_img(self, valor):
        bri_image = self.image.copy()
        bri_image = bri_image + valor
        plt.imshow(bri_image)
        plt.show()
        return bri_image
    
    # Funcion para aplicar convolucion (checar notas del 14 de Octubre)
    def con_img(self, kernel=np.ones((3,3))):
        image = self.image.copy()
        m, n = image.shape
        result = np.zeros((m-1,n-1))
        for i in range(m-1):
            for j in range(n-1):
                section = image[i:i+3, j:j+3]
                # print(f'start, i = {i} and j = {j}')
                value = 0
                for a, fila in enumerate(section):
                    for b, valor in enumerate(fila):
                        value += valor * kernel[a][b]
                result[i, j] = round(value/9)
                # print(result[i,j])
        '''m, n = result.shape
        print(f'Ancho:  {n}, Alto:  {m}')'''
        return result
        """plt.imshow(result)
        plt.show()"""

            # TODO: AL parecer esto es gran parte de lo de la tarea pasada, solo falta que el kernel que aplico detecte bordes

    # Funcion para identificar bordes de imagen.
    def bor_img(self):
        self.image = self.bnw_img()
        hor_ker = [[-1, -1, -1], 
                   [2, 2, 2],
                   [-1, -1, -1]]
        
        ver_ker = [[-1, 2, -1],
                   [-1, 2, -1],
                   [-1, 2, -1]]
        
        two_ker = [[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]]
        
        hor_image = self.con_img(hor_ker)
        ver_image = self.con_img(ver_ker)
        two_image = self.con_img(two_ker)
        plt.figure(figsize=(12, 6))
    
        plt.subplot(2, 2, 1)  
        plt.imshow(self.image, cmap='gray')
        plt.title('Imagen Original')
        
        plt.subplot(2, 2, 2)
        plt.imshow(hor_image, cmap='gray')
        plt.title('Bordes Horizontales')
        
        plt.subplot(2, 2, 3)
        plt.imshow(ver_image, cmap='gray')
        plt.title('Bordes Verticales')
        
        plt.subplot(2, 2, 4)
        plt.imshow(two_image, cmap='gray')
        plt.title('Bordes Bordes')
        
        plt.tight_layout()

        plt.show()


imgm = ImageManipulation(image)
imgm.bor_img()

""" image = cv2.imread('lena.tif')
cv2.imshow('Lena', image) 
cv2.waitKey(0)
cv2.destroyAllWindows() """
