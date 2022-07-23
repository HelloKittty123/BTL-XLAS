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
from skimage import img_as_float

# Xây dựng phép tích chập và phép tương quan chéo
def convolute(image, kernel):
  # Lật ma trận mặt nạ
  r,c = kernel.shape
  flipped_kernel = np.zeros((r,c),np.float32)
  h=0
  # Ma trận có số hàng và cột lẻ thì lật ngược ma trận 180 độ
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
        if (len(entry2.get()) and int(entry2.get())%2 == 1):
            medianFilt[:, :, i] = medianFilter(img[:, :, i], int(entry2.get()))
        else:
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
        if (len(entry2.get()) and int(entry2.get())%2 == 1):
            mean[:, :, i] = meanFilter(img[:, :, i], int(entry2.get()))
            mean_corr[:, :, i] = meanFilter(img[:, :, i], int(entry2.get()), 0)
        else:
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
        if (len(entry2.get()) and int(entry2.get())%2 == 1):
            GaussianFilt[:, :, i] = GaussianFilter(img[:, :, i], int(entry2.get()), math.sqrt(2))
            GaussianFilt_corr[:, :, i] = GaussianFilter(img[:, :, i], int(entry2.get()), math.sqrt(2), 0)
        else:
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
# Bộ lọc Unsharp mask
def unsharpMask():
    filepath = filedialog.askopenfilename(title="Open file okay?", )
    img = cv2.imread(filepath)
    # entry1.delete(0, END)
    # entry1.insert(0, str(filepath))
    image = img_as_float(img)
    gaussian_3 = cv2.GaussianBlur(image, (0, 0), 2.0)
    unsharp_image = image + 2 * (image - gaussian_3)  # K =2
    unsharp_image_2 = image * 2 - gaussian_3  # k = 1
    # unsharp = cv2.addWeighted(image, 2.0, gaussian_3, -1.0, 0)
    # print(unsharp_image_2[320:350,90:110])
    # print("/////////")
    # print(unsharp[320:350,90:110])
    # cv2.imwrite("example_unsharp.jpg", unsharp_image)
    plt.figure(figsize=(15, 15))
    plt.subplot(131), plt.title('Original'), plt.imshow(image[:, :, ::-1])
    plt.subplot(132), plt.title('Unsharp mask k = 2'), plt.imshow(unsharp_image[:, :, ::-1])
    plt.subplot(133), plt.title('Unsharp mask k = 1'), plt.imshow(unsharp_image_2[:, :, ::-1])
    # plt.subplot(144), plt.title('Mean v Correlation'), plt.imshow(unsharp[:,:,::-1])
    plt.show()
#----------------------------------------
# Bộ lọc Laplacian
# Xây dựng bộ lọc Laplacian
def LaplaceFilter(image,kernel_type = 1,convol=1):
  #Tạo kernel
  if kernel_type==2:
    kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
  elif kernel_type==3:
    kernel = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]])
  else:
    kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
  #Tích chập hoặc tương quan
  if convol == 1:
    output = convolute(image,kernel)
  else:
    output = cv2.filter2D(image,-1,kernel,borderType=cv2.BORDER_ISOLATED)
  return output


# Test bộ lọc Laplacian
def testLaplacian():
    filepath = filedialog.askopenfilename(title="Open file okay?", )
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    entry1.delete(0, END)
    entry1.insert(0, str(filepath))
    Laplacian = cv2.Laplacian(img,cv2.CV_64F)
    LaplacianFilt1 = LaplaceFilter(img)
    LaplacianFilt2 = LaplaceFilter(img,2)
    LaplacianFilt3 = LaplaceFilter(img,3)
    LaplacianFilt1_corr = LaplaceFilter(img,1,0)
    LaplacianFilt2_corr = LaplaceFilter(img,2,0)
    LaplacianFilt3_corr = LaplaceFilter(img,3,0)

    cv2.imwrite('girl_gray.jpg', img)
    cv2.imwrite('Laplacian.jpg', Laplacian)
    cv2.imwrite('Laplacian1.jpg',LaplacianFilt1)
    cv2.imwrite('Laplacian2.jpg', LaplacianFilt2)
    cv2.imwrite('Laplacian3.jpg', LaplacianFilt3)
    cv2.imwrite('Laplacian1_corr.jpg',LaplacianFilt1_corr)
    cv2.imwrite('Laplacian2_corr.jpg', LaplacianFilt2_corr)
    cv2.imwrite('Laplacian3_corr.jpg', LaplacianFilt3_corr)

    plt.figure(figsize=(18,18))
    plt.subplot(421), plt.title('Original'), plt.imshow(img,cmap='gray',vmin=0,vmax=255)
    plt.subplot(422), plt.title('Laplacian'), plt.imshow(Laplacian,cmap='gray',vmin=0,vmax=255)
    plt.subplot(423), plt.title('LaplacianFilt type1'), plt.imshow(LaplacianFilt1,cmap='gray',vmin=0,vmax=255)
    plt.subplot(425), plt.title('LaplacianFilt type2'), plt.imshow(LaplacianFilt2,cmap='gray',vmin=0,vmax=255)
    plt.subplot(427), plt.title('LaplacianFilt type3'), plt.imshow(LaplacianFilt3,cmap='gray',vmin=0,vmax=255)
    plt.subplot(424), plt.title('LaplacianFilt type1 correlation'), plt.imshow(LaplacianFilt1_corr,cmap='gray',vmin=0,vmax=255)
    plt.subplot(426), plt.title('LaplacianFilt type2 correlation'), plt.imshow(LaplacianFilt2_corr,cmap='gray',vmin=0,vmax=255)
    plt.subplot(428), plt.title('LaplacianFilt type3 correlation'), plt.imshow(LaplacianFilt3_corr,cmap='gray',vmin=0,vmax=255)
    plt.show()
#----------------------------------------
# Bộ lọc Sobel
def SobelFilter(image,gauss_ksize=5,dx=1,dy=1,convol=1,threshold=60):
  GaussKernel = createGaussianKernel(gauss_ksize,1)
  X = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
  Y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    #Tích chập hoặc tương quan
  if convol == 1:
      kernel_x = convolute(GaussKernel,X)
      sobel_x = convolute(image,kernel_x)
      kernel_y = convolute(GaussKernel,Y)
      sobel_y = convolute(image,kernel_y)
  else:
      kernel_x = cv2.filter2D(GaussKernel,cv2.CV_64F,X,borderType=cv2.BORDER_ISOLATED)
      sobel_x = cv2.filter2D(image,cv2.CV_64F,kernel_x,borderType=cv2.BORDER_ISOLATED)
      kernel_y = cv2.filter2D(GaussKernel,cv2.CV_64F,Y,borderType=cv2.BORDER_ISOLATED)
      sobel_y = cv2.filter2D(image,cv2.CV_64F,kernel_y,borderType=cv2.BORDER_ISOLATED)

  if dx==1 and dy==0:
    return sobel_x
  if dx==0 and dy==1:
    return sobel_y
  sobel = np.sqrt(np.square(sobel_x)+np.square(sobel_y))
  img_sobel = np.uint8(sobel)
  for i in range(img_sobel.shape[0]):
      for j in range(img_sobel.shape[1]):
          if img_sobel[i][j] < threshold:
              img_sobel[i][j] = 0
  return img_sobel

# Test bộ lọc Sobel
def testSobel():
    filepath = filedialog.askopenfilename(title="Open file okay?", )
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    entry1.delete(0, END)
    entry1.insert(0, str(filepath))
    Sobelx = cv2.Sobel(img,cv2.CV_64F,1,0)
    Sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
    Sobel = cv2.Sobel(img,cv2.CV_64F,1,1)
    SobelFiltx = SobelFilter(img,5,1,0)
    SobelFilty = SobelFilter(img,5,0,1)
    SobelFilt = SobelFilter(img,5,1,1)
    SobelFiltx_corr = SobelFilter(img,5,1,0,0)
    SobelFilty_corr = SobelFilter(img,5,0,1,0)
    SobelFilt_corr = SobelFilter(img,5,1,1,0)

    cv2.imwrite('Sobelx.jpg', Sobelx)
    cv2.imwrite('Sobely.jpg', Sobely)
    cv2.imwrite('Sobel.jpg', Sobel)
    cv2.imwrite('SobelFiltx.jpg',SobelFiltx)
    cv2.imwrite('SobelFilty.jpg',SobelFilty)
    cv2.imwrite('SobelFilt.jpg',SobelFilt)
    cv2.imwrite('SobelFiltx_corr.jpg',SobelFiltx_corr)
    cv2.imwrite('SobelFilty_corr.jpg',SobelFilty_corr)
    cv2.imwrite('SobelFilt_corr.jpg',SobelFilt_corr)

    plt.figure(figsize=(18,18))
    plt.subplot(431), plt.title('Original'), plt.imshow(img,cmap='gray',vmin=0,vmax=255)
    plt.subplot(434), plt.title('Sobel'), plt.imshow(Sobel,cmap='gray',vmin=0,vmax=255)
    plt.subplot(435), plt.title('SobelFilter'), plt.imshow(SobelFilt,cmap='gray',vmin=0,vmax=255)
    plt.subplot(436), plt.title('SobelFilter correlation'), plt.imshow(SobelFilt_corr,cmap='gray',vmin=0,vmax=255)
    plt.subplot(437), plt.title('Sobel x'), plt.imshow(Sobelx,cmap='gray',vmin=0,vmax=255)
    plt.subplot(438), plt.title('SobelFilter x'), plt.imshow(SobelFiltx,cmap='gray',vmin=0,vmax=255)
    plt.subplot(439), plt.title('SobelFilter x correlation'), plt.imshow(SobelFiltx_corr,cmap='gray',vmin=0,vmax=255)
    plt.subplot(4,3,10), plt.title('Sobel y'), plt.imshow(Sobely,cmap='gray',vmin=0,vmax=255)
    plt.subplot(4,3,11), plt.title('SobelFilter y'), plt.imshow(SobelFilty,cmap='gray',vmin=0,vmax=255)
    plt.subplot(4,3,12), plt.title('SobelFilter y correlation'), plt.imshow(SobelFilty_corr,cmap='gray',vmin=0,vmax=255)
    plt.show()
#----------------------------------------

# Bộ lọc hiệu ứng mùa đông
def addSnow(image):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    image_HLS = np.array(image_HLS, dtype = np.float64)
    brightness_coefficient = 2.5
    snow_point=140 ## increase this for more snow
    image_HLS[:,:,1][image_HLS[:,:,1]<snow_point] = image_HLS[:,:,1][image_HLS[:,:,1]<snow_point]*brightness_coefficient ## scale pixel values up for channel 1(Lightness)
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return image_RGB

# Test bộ lọc hiệu ứng mùa đông
def testAddSnow():
    filepath = filedialog.askopenfilename(title="Open file okay?", )
    img = cv2.imread(filepath)
    entry1.delete(0, END)
    entry1.insert(0, str(filepath))
    img_cop = addSnow(img)
    cv2.imwrite('SnowFilter.jpg', img_cop)
    plt.figure(figsize=(10,10))
    plt.subplot(221), plt.title('Original'), plt.imshow(img[:,:,::-1])
    plt.subplot(222), plt.title('SnowFilter'), plt.imshow(img_cop[:,:,::-1])
    plt.show()
#----------------------------------------

#Bộ lọc DFT
def testDFT():
    filepath = filedialog.askopenfilename(title="Open file okay?", )
    img = cv2.imread(filepath, 0)
    entry1.delete(0, END)
    entry1.insert(0, str(filepath))
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.ones((rows, cols, 2), np.uint8)
    r = 80
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0

    # apply mask and inverse DFT
    fshift = dft_shift * mask

    fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(img, cmap='gray')
    ax1.title.set_text('Input Image')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(magnitude_spectrum, cmap='gray')
    ax2.title.set_text('FFT of image')
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(fshift_mask_mag, cmap='gray')
    ax3.title.set_text('FFT + Mask')
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(img_back, cmap='gray')
    ax4.title.set_text('After inverse FFT')
    plt.show()
#----------------------------------------
# Ứng dụng
root = Tk()
root.title("BÀI TẬP LỚN XỬ LÝ ẢNH")
root.geometry("700x550")
logo = Image.open(r'./3.jpg')
logo = ImageTk.PhotoImage(logo)
logo_lb = tk.Label(image=logo)
logo_lb.image = logo
logo_lb.grid(column = 1 , row = 1)
button = Button(text="Bộ lọc trung vị",bg = 'pink',fg = 'white',font = 'Times 20 bold',borderwidth= 0, command=testMedianFilter)
button.place(x=50,y=50)
button1 = Button(text="Bộ lọc trung bình",bg = 'pink',fg = 'white',font = 'Times 20 bold',borderwidth= 0, command=testMeanFilter)
button1.place(x=50,y=150)
button2 = Button(text="Bộ lọc Gaussian",bg = 'pink',fg = 'white',font = 'Times 20 bold',borderwidth= 0, command=testGaussianFilter)
button2.place(x=50,y=250)
button3 = Button(text="Bộ lọc Unsharp Mask",bg = 'pink',fg = 'white',font = 'Times 20 bold',borderwidth= 0, command=unsharpMask)
button3.place(x=350,y=50)
button4 = Button(text="Bộ lọc Laplacian",bg = 'pink',fg = 'white',font = 'Times 20 bold',borderwidth= 0, command=testLaplacian)
button4.place(x=350,y=150)
button5 = Button(text="Bộ lọc Sobel",bg = 'pink',fg = 'white',font = 'Times 20 bold',borderwidth= 0, command=testSobel)
button5.place(x=350,y=250)
button6 = Button(text="Bộ lọc tạo hiệu ứng mùa đông",bg = 'pink',fg = 'white',font = 'Times 20 bold',borderwidth= 0, command=testAddSnow)
button6.place(x=50,y=350)
button7 = Button(text="Bộ lọc DFT",bg = 'pink',fg = 'white',font = 'Times 20 bold',borderwidth= 0, command=testDFT)
button7.place(x=450,y=350)
lbl1 = Label(root, text = "Path File:")
lbl1.place(x=40, y=450)
entry1 = Entry(root,font=("Times New Roman",14) ,width = 50)
entry1.place(x = 100 , y =450)
lbl2 = Label(root, text = "Kernel size:")
lbl2.place(x=40, y=500)
entry2 = Entry(root,font=("Times New Roman",14) ,width = 50)
entry2.place(x = 110 , y =500)


root.mainloop()