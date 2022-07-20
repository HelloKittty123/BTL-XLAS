from tkinter import *
import tkinter as tk
from tkinter import filedialog
import cv2
import os
from PIL import Image, ImageTk
from tkinter import filedialog

import numpy as np
from scipy import signal
import cv2
from matplotlib import pyplot as plt
import math

# Xây dựng phép tích chập và phép tương quan chéo
def convolute(image, kernel):
  # Lật ma trận mặt nạ
  r,c = kernel.shape
  flipped_kernel = np.zeros((r,c),np.float32)
  h=0
  if r==c and r%2 :
    h = int((r-1)/2)
    for i in range(-h,h+1):
     for j in range(-h,h+1):
      flipped_kernel[i+h,j+h]=kernel[-i+h,-j+h]
  # Thêm padding cho ảnh đầu vào
  m,n = image.shape
  padded_image = np.zeros((m+2*h,n+2*h),np.float32)
  padded_image[h:-h,h:-h] = image
  # Tạo ảnh đầu ra
  output = np.zeros((m,n),np.float32)
  for i in range(m):
   for j in range(n):
    output[i,j]=(flipped_kernel * padded_image[i: i+r, j: j+r]).sum()
  return output

def cross_correlate(image, kernel):
  r,c = kernel.shape
  h=0
  if r==c and r%2 :
    h = int((r-1)/2)
  # Thêm padding cho ảnh đầu vào
  m,n = image.shape
  padded_image = np.zeros((m+2*h,n+2*h),int)
  padded_image[h:-h,h:-h] = image
  # Tạo ảnh đầu ra
  output = np.zeros((m,n),np.float32)
  for i in range(m):
   for j in range(n):
    output[i,j]=(kernel * padded_image[i: i+r, j: j+r]).sum()
  return output
#----------------------------------------
# Xây dựng bộ lọc trung vị
def medianFilter(image, kernel_size):
 h=kernel_size//2
 m,n = image.shape
 # Thêm padding cho ảnh đầu vào
 padded_image = np.zeros((m+2*h,n+2*h),int)
 padded_image[h:-h,h:-h] = image
 padded_image[:h,h:-h] = image[:h,:]
 padded_image[-h:,h:-h] = image[-h:,:]
 padded_image[h:-h,:h] = image[:,:h]
 padded_image[h:-h,-h:] = image[:,-h:]
 Kernel_len = kernel_size*kernel_size
 # Tạo ma trận đầu ra
 output = np.zeros((m,n),int)
 for i in range(m):
   for j in range(n):
     array = np.zeros(Kernel_len,int)
     for p in range(kernel_size):
       for q in range(kernel_size):
         array[p*kernel_size+q]=padded_image[i+p,j+q]
     array = np.sort(array)
     output[i,j]=array[len(array)//2]
 return output

# Test bộ lọc trung vị
def testMedianFilter():
    filepath = filedialog.askopenfilename(title="Open file okay?", )
    img = cv2.imread(filepath)
    entry1.delete(0, END)
    entry1.insert(0, str(filepath))
    medianFilt = np.copy(img)
    for i in range(3):
     medianFilt[:,:,i]=medianFilter(img[:,:,i],5)

    cv2.imwrite('medianFilter.jpg',medianFilt)
    plt.figure(figsize=(18,18))
    plt.subplot(131), plt.title('Original'), plt.imshow(img[:,:,::-1])
    plt.subplot(132), plt.title('Median'), plt.imshow(medianFilt[:,:,::-1])
    plt.show()
#----------------------------------------
# Xây dựng bộ lọc trung bình
def meanFilter(image, kernel_size, convol=1):
    # Tạo kernel
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    # Tích chập hoặc tương quan
    if convol == 1:
        output = convolute(image, kernel)
    else:
        # output = cv2.filter2D(image,-1,kernel,borderType=cv2.BORDER_ISOLATED)
        output = cross_correlate(image, kernel)
    return output

# Test bộ lọc trung bình
def testMeanFilter():
    filepath = filedialog.askopenfilename(title="Open file okay?", )
    img = cv2.imread(filepath)
    entry1.delete(0, END)
    entry1.insert(0, str(filepath))
    mean = mean_corr = np.copy(img)
    for i in range(3):
        mean[:,:,i] = meanFilter(img[:,:,i],5)
        mean_corr[:,:,i] = meanFilter(img[:,:,i],5,0)

    cv2.imwrite('mean.jpg', mean)
    cv2.imwrite('mean_corr.jpg', mean)

    plt.figure(figsize=(10,10))
    plt.subplot(221), plt.title('Original'), plt.imshow(img[:,:,::-1])
    plt.subplot(222), plt.title('Mean'), plt.imshow(mean[:,:,::-1])
    plt.subplot(223), plt.title('Mean v Correlation'), plt.imshow(mean[:,:,::-1])
    plt.show()
#----------------------------------------
# Xây dựng bộ lọc Gaussian
def createGaussianKernel(kernel_size,sigma):
  h = kernel_size//2
  var = np.square(sigma)
  kernel=np.zeros((kernel_size,kernel_size),np.float)
  for i in range(kernel_size):
    for j in range(kernel_size):
      kernel[i,j] = np.exp(-(np.square(i-h)+np.square(j-h))/2/var)
  # print(kernel)
  #a = np.amin(kernel)
  #kernel = kernel*(1/a)
  # print(kernel)
  #print(kernel.sum())
  kernel = kernel/kernel.sum()
  #print(kernel)
  return kernel

def GaussianFilter(image,kernel_size,sigma,convol=1):
  #Tạo kernel
  kernel = createGaussianKernel(kernel_size,sigma)
    #Tích chập hoặc tương quan
  if convol == 1:
    output = convolute(image,kernel)
  else:
    output = cv2.filter2D(image,-1,kernel,borderType=cv2.BORDER_ISOLATED)
  return output

# Test bộ lọc Gaussian
def testGaussianFilter():
    filepath = filedialog.askopenfilename(title="Open file okay?", )
    img = cv2.imread(filepath)
    entry1.delete(0, END)
    entry1.insert(0, str(filepath))
    GaussianFilt = GaussianFilt_corr = np.copy(img)
    for i in range(3):
        GaussianFilt[:,:,i] = GaussianFilter(img[:,:,i],7,math.sqrt(2))
        GaussianFilt_corr[:,:,i] = GaussianFilter(img[:,:,i],7,math.sqrt(2),0)

    cv2.imwrite('GaussianFilter.jpg', GaussianFilt)
    cv2.imwrite('GaussianFilter_corr.jpg', GaussianFilt_corr)

    print('Kernel được tạo ra: \n',createGaussianKernel(5,math.sqrt(2)))
    plt.figure(figsize=(10,10))
    #open cv ảnh vào là BGR, show ra phải RGB : plot hoặc io, cv2 thì k cần đổi
    plt.subplot(221), plt.title('Original'), plt.imshow(img[:,:,::-1])
    plt.subplot(222), plt.title('GaussianFilter'), plt.imshow(GaussianFilt[:,:,::-1])
    plt.subplot(223), plt.title('GaussianFilter v Correlation'), plt.imshow(GaussianFilt_corr[:,:,::-1])
    plt.show()
#----------------------------------------

# Ứng dụng
root = Tk()
root.title("BÀI TẬP LỚN XỬ LÝ ẢNH")
root.geometry("660x550")
logo = Image.open(r'./screen.jpg')
logo = ImageTk.PhotoImage(logo)
logo_lb = tk.Label(image=logo)
logo_lb.image = logo
logo_lb.grid(column = 1 , row = 1)
button = Button(text="Bộ lọc trung vị",bg = 'pink',fg = 'white',font = 'Times 20 bold',borderwidth= 0, command=testMedianFilter)
button.place(x=250,y=50)
button1 = Button(text="Bộ lọc trung bình",bg = 'pink',fg = 'white',font = 'Times 20 bold',borderwidth= 0, command=testMeanFilter)
button1.place(x=240,y=150)
button2 = Button(text="Bộ lọc Gaussian",bg = 'pink',fg = 'white',font = 'Times 20 bold',borderwidth= 0, command=testGaussianFilter)
button2.place(x=245,y=250)
entry1 = Entry(root,font=("Times New Roman",14) ,width = 50)
entry1.place(x = 100 , y =350)

root.mainloop()