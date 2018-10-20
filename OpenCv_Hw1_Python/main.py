import cv2
import numpy as np
import tkinter as tk
import PIL.Image, PIL.ImageTk

def fun1_1():
  dog = cv2.imread("../img/dog.bmp", cv2.IMREAD_COLOR)
  print('Height:', dog.shape[0])
  print('width:', dog.shape[1])
  cv2.imshow('1',dog)

def fun1_2():
  img = cv2.imread( '../img/color.png', cv2.IMREAD_COLOR )
  rbg = img[...,[1,2,0]]
  cv2.imshow('original', img)
  cv2.imshow('color conversion', rbg)
def fun1_3():
  dog = cv2.imread( '../img/dog.bmp', cv2.IMREAD_COLOR )
  dog__f = cv2.flip(dog, 1)
  cv2.imshow('Flip',dog__f)
def fun1_4():
  #load image
  def Change(x):
      alpha = cv2.getTrackbarPos('Blend', 'Blending')/100
      dst = cv2.addWeighted(img1,alpha,img2,1-alpha,0)
      cv2.imshow('Blending',dst)
  img1 = cv2.imread( '../img/dog.bmp', cv2.IMREAD_COLOR )
  img2 = cv2.flip(img1, 1)
  cv2.namedWindow('Blending')
  cv2.createTrackbar('Blend', 'Blending', 0,100,Change)
  # default image : flipped image
  dst = cv2.addWeighted(img1,0,img2,1,0)
  cv2.imshow('Blending',dst)

def fun2_1():
  def SobelOperator(roi,operator_type):
    if operator_type == "horizontal":
        sobel_operator = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    elif operator_type == "vertical":
        sobel_operator = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    else:
        raise("type Error")
    result = np.abs(np.sum(roi*sobel_operator))
    return result
  def SobelAlogrithm(image,operator_type):
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_DEFAULT)
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            new_image[i-1,j-1] = SobelOperator(image[i-1:i+2,j-1:j+2],operator_type)
    new_image = new_image*(255/np.max(image))
    return new_image.astype(np.uint8)

  screw = cv2.imread("../img/M8.jpg", cv2.IMREAD_COLOR)
  grayImg = cv2.cvtColor(screw, cv2.COLOR_BGR2GRAY)
  blurImg = cv2.GaussianBlur(grayImg,(3,3),0)
  cv2.imshow('Gray and Smooth',blurImg)
  re = SobelAlogrithm(blurImg,"horizontal")
  ret, result_H = cv2.threshold(re, 40, 255, cv2.THRESH_BINARY)
  cv2.imshow('tt', result_H)
  re1 = SobelAlogrithm(blurImg,"vertical")
  ret, result_V = cv2.threshold(re1, 40, 255, cv2.THRESH_BINARY)
  cv2.imshow('ttV', result_V)

  def Change_M(x):
      thre = cv2.getTrackbarPos('magnitude', 'Magnitude')
      dst = cv2.addWeighted(re,0.5,re1,0.5,0)
      ret, dst = cv2.threshold(dst, thre, 255, cv2.THRESH_BINARY)
      cv2.imshow('Magnitude',dst)
  cv2.namedWindow('Magnitude')
  cv2.createTrackbar('magnitude', 'Magnitude', 40,255,Change_M)
  dst = cv2.addWeighted(re,0.5,re1,0.5,0)
  ret, dst = cv2.threshold(dst, 40, 255, cv2.THRESH_BINARY)
  cv2.imshow('Magnitude',dst)

  def Change_A(x):
    newImg = np.zeros(blurImg.shape)
    angle = cv2.getTrackbarPos('angle', 'Direction')
    for i in range(0,blurImg.shape[0]-2):
      for j in range(0,blurImg.shape[1]-2):
        g = int(re[i][j]) + int(re1[i][j])
        if g>255:
          g = 255

        rad = np.arctan2( int(re[i][j]), int(re1[i][j]) )
        rad = float(rad) + np.pi
        theta = (rad / np.pi) * 180

        if(theta<=float(angle)+10 and theta>=float(angle)-10):
          newImg[i][j] = g
        else:
          newImg[i][j] = 0
    cv2.imshow('Direction',newImg)

  cv2.namedWindow('Direction')
  cv2.createTrackbar('angle', 'Direction', 10,360,Change_A)
  imgA = cv2.addWeighted(re,0.5,re1,0.5,0)
  cv2.imshow('Direction',imgA)
  for i in range(0,re.shape[0]-1):
    for j in range(0,re.shape[1]-1):
      if(re[i][j]>255):
        print(re[i-1][j-1])


  # kernelX = [
  #       [ -1, 0, 1 ],
  #       [ -2, 0, 2 ],
  #       [ -1, 0, 1 ]
  #   ]

  # kernelY = [
  #     [ -1, -2, -1 ],
  #     [  0,  0,  0 ],
  #     [  1,  2,  1 ]
  # ]

  # kernelXMat = np.array(kernelX, dtype=float)
  # kernelYMat = np.array(kernelY, dtype=float)

  # horizGradient = cv2.filter2D(blurImg, cv2.CV_32F, kernelXMat)
  # vertGradient = cv2.filter2D(blurImg, cv2.CV_32F, kernelYMat)
  # print(horizGradient)

  # abs_horizGradient = cv2.convertScaleAbs(horizGradient)
  # abs_vertGradient = cv2.convertScaleAbs(vertGradient)


  # ret, result_H = cv2.threshold(abs_horizGradient, 40, 255, cv2.THRESH_BINARY)
  # ret, result_V = cv2.threshold(abs_vertGradient, 40, 255, cv2.THRESH_BINARY)

  # cv2.imshow('horizontal edge',result_H)
  # cv2.imshow('vertical edge',result_V)

  # def Change_M(x):
  #     thre = cv2.getTrackbarPos('magnitude', 'Magnitude')
  #     dst = cv2.addWeighted(abs_horizGradient,0.5,abs_vertGradient,0.5,0)
  #     ret, dst = cv2.threshold(dst, thre, 255, cv2.THRESH_BINARY)
  #     cv2.imshow('Magnitude',dst)
  # cv2.namedWindow('Magnitude')
  # cv2.createTrackbar('magnitude', 'Magnitude', 40,255,Change_M)
  # dst = cv2.addWeighted(abs_horizGradient,0.5,abs_vertGradient,0.5,0)
  # ret, dst = cv2.threshold(dst, 40, 255, cv2.THRESH_BINARY)
  # cv2.imshow('Magnitude',dst)

  # def Change_A(x):
  #     thre = cv2.getTrackbarPos('angle', 'Direction')
  # cv2.namedWindow('Direction')
  # cv2.createTrackbar('angle', 'Direction', 10,360,Change_A)
  # imgA = cv2.addWeighted(abs_horizGradient,0.5,abs_vertGradient,0.5,0)
  # cv2.imshow('Direction',imgA)


def fun3_1():
  pyramids = cv2.imread("../img/pyramids_Gray.jpg", cv2.IMREAD_COLOR)
  # pyrDown to get Gaussian pyramid
  GauPyra_1 = cv2.pyrDown(pyramids)
  GauPyra_2 = cv2.pyrDown(GauPyra_1)

  # use Gaussian pyramid minus pyrUp(Gaussian pyramid)
  # to get Laplacian pyramid
  LapPyra_1 = cv2.subtract( GauPyra_1, cv2.pyrUp( GauPyra_2 ) )
  LapPyra_0 = cv2.subtract( pyramids, cv2.pyrUp( GauPyra_1 ) )

  # use Laplacian pyramid plus pyrUp(Inverse pyramid)
  # to get Inverse pyramid
  InvPyra_1 = cv2.add( LapPyra_1, cv2.pyrUp(GauPyra_2) )
  InvPyra_0 = cv2.add( LapPyra_0, cv2.pyrUp(InvPyra_1) )

  # test = cv2.Laplacian(pyramids,cv2.CV_8U)
  # cv2.imshow('test', test)
  # cv2.imshow('origin', pyramids)
  # cv2.imshow('Laplacian pyramid level 1 ', LapPyra_1)
  # cv2.imshow('Inverse pyramid level 1 ', GauPyra_1)
  # cv2.imshow('Laplacian pyramid level 1 ', LapPyra_1)
  # cv2.imshow('origin', pyramids)
  cv2.imshow('Gaussian pyramid level 1', GauPyra_1)
  cv2.imshow('Laplacian pyramid level 0 ', LapPyra_0)
  cv2.imshow('Inverse pyramid level 0 ', InvPyra_0)
  cv2.imshow('Inverse pyramid level 1 ', InvPyra_1)

def fun4_1():
  QR = cv2.imread("../img/QR.png", cv2.IMREAD_COLOR)

  # get gray scale image
  grayImg = cv2.cvtColor(QR, cv2.COLOR_BGR2GRAY)

  # setting global threshold
  ret, result = cv2.threshold(grayImg,80,255,cv2.THRESH_BINARY)

  cv2.imshow('Original', QR)
  cv2.imshow('Threshold', result)
def fun4_2():
  QR = cv2.imread("../img/QR.png", cv2.IMREAD_COLOR)

  # get gray scale and blurred image
  grayImg = cv2.cvtColor(QR, cv2.COLOR_BGR2GRAY)
  grayImg = cv2.medianBlur(grayImg,5)

  # setting local threshold
  result = cv2.adaptiveThreshold(grayImg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,19,-1)

  cv2.imshow('Original', QR)
  cv2.imshow('Adapted Threshold', result)
def fun5_1():
  # get the value from ui
  Angle = float(e_angle.get())
  Scale = float(e_scale.get())
  Tx = float(e_tx.get())
  Ty = float(e_ty.get())

  # read image
  img = cv2.imread('../img/OriginalTransform.png')

  # making translation matrix
  H = np.float32([[1,0,Tx],[0,1,Ty]])

  # translate the image
  rows,cols = img.shape[:2]
  tansImg = cv2.warpAffine(img,H,(rows,cols))

  # making rotate and scale matrix
  rows,cols = tansImg.shape[:2]
  M = cv2.getRotationMatrix2D((130+Tx,125+Ty),Angle,Scale)

  # rotating and Scaling the image
  result = cv2.warpAffine(tansImg,M,(rows,cols))

  cv2.imshow(', Scaling, Translation',img)
  cv2.imshow('Rotation, Scaling, Translation',result)
def fun5_2():
  imgPoint=[]
  objPoint = [[0,0],[450,0],[450,450],[0,450]]
  def getPoint(event,x,y,flags,param):
      # if click the button, putting that point into the array imgPoint
      # show a red spot on that point
      if event == cv2.EVENT_LBUTTONDOWN:
          nonlocal imgPoint
          imgPoint.append([x,y])
          if(len(imgPoint) <= 4):
              cv2.circle(img,(x,y), 10, (0,0,255), -1)
              cv2.imshow('Original',img)

          # if get exactly four point
          # doing the perspective function
          if len(imgPoint)==4:
              imgP = np.float32(imgPoint)
              objP = np.float32(objPoint)
              M = cv2.getPerspectiveTransform(imgP,objP)
              dst = cv2.warpPerspective(perspect,M,(450,450))

              cv2.imshow('Perspective image', dst)
              print(imgPoint)

  img = cv2.imread('../img/OriginalPerspective.png')
  perspect = img.copy()
  cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
  cv2.imshow('Original',img)
  cv2.setMouseCallback('Original',getPoint)


main_window = tk.Tk()
main_window.geometry('1080x720')
main_window.title('HW1')

# all frames
left_frame = tk.Frame(main_window)
left_frame.place(relwidth=0.3, relheight=1)
middle_frame = tk.Frame(main_window)
middle_frame.place(relx=0.3, relwidth=0.3, relheight=1)
right_frame = tk.Frame(main_window)
right_frame.place(relx=0.6, relwidth=0.4, relheight=1)
lf51 = tk.LabelFrame(right_frame, text='5.1', font=("Helvetica", 20))
lf51.grid(row=1, columnspan=7, sticky='nsew', padx=5, ipadx=5, ipady=10)
lfpara = tk.LabelFrame(lf51, text='Parameters', font=("Helvetica", 20))
lfpara.grid(row=0, sticky='nsew', padx=5, ipadx=5, ipady=10)
lfparas = tk.LabelFrame(lfpara, text='')
lfparas.grid(row=0, sticky='nsew', padx=5, ipadx=20, ipady=25)

# all labels
l1 = tk.Label(left_frame, text='1. Image Processing', font=("Helvetica", 24))
l1.grid(row=0, sticky='w', padx=5, pady=30)
l2 = tk.Label(left_frame, text='2. Edge Detection', font=("Helvetica", 24))
l2.grid(row=5, sticky='w', padx=5, pady=30)
l3 = tk.Label(middle_frame, text='3. Image Pyramids', font=("Helvetica", 24))
l3.grid(row=0, sticky='w', padx=5, pady=30)
l4 = tk.Label(middle_frame, text='4. Adaptive Threshhold', font=("Helvetica", 24))
l4.grid(row=2, sticky='w', padx=5, pady=30)
l5 = tk.Label(right_frame, text='5. Image Transformation', font=("Helvetica", 24))
l5.grid(row=0, sticky='w', padx=5, pady=30)
l_angle = tk.Label(lfparas, text='Angle:    ')
l_angle.grid(row=0, column=0, sticky='nsew', pady=10)
l_angle_unit = tk.Label(lfparas, text='deg')
l_angle_unit.grid(row=0, column=2, sticky='nsew', pady=10)
l_scale = tk.Label(lfparas, text='Scale:    ')
l_scale.grid(row=1, column=0, sticky='nsew', pady=10)
l_tx = tk.Label(lfparas, text='Tx:    ')
l_tx.grid(row=2, column=0, sticky='nsew', pady=10)
l_tx_unit = tk.Label(lfparas, text='pixel')
l_tx_unit.grid(row=2, column=2, sticky='nsew', pady=10)
l_ty = tk.Label(lfparas, text='Ty:    ')
l_ty.grid(row=3, column=0, sticky='nsew', pady=10)
l_ty_unit = tk.Label(lfparas, text='pixel')
l_ty_unit.grid(row=3, column=2, sticky='nsew', pady=10)

# all buttons
b11 = tk.Button(left_frame, text='1.1 Load Image', font=("Helvetica", 18), command=fun1_1)
b11.grid(row=1, sticky='nsew', padx=20, pady=20)
b12 = tk.Button(left_frame, text='1.2 Color Coversion', font=("Helvetica", 18), command=fun1_2)
b12.grid(row=2, sticky='nsew', padx=20, pady=20)
b13 = tk.Button(left_frame, text='1.3 Image Flipping', font=("Helvetica", 18), command=fun1_3)
b13.grid(row=3, sticky='nsew', padx=20, pady=20)
b14 = tk.Button(left_frame, text='1.4 Blending', font=("Helvetica", 18), command=fun1_4)
b14.grid(row=4, sticky='nsew', padx=20, pady=20)
b21 = tk.Button(left_frame, text='2.1 Edge Detection', font=("Helvetica", 18), command=fun2_1)
b21.grid(row=6, sticky='nsew', padx=20, pady=20)
b31 = tk.Button(middle_frame, text='3.1 Image Pyramids', font=("Helvetica", 18), command=fun3_1)
b31.grid(row=1, sticky='nsew', padx=20, pady=20)
b41 = tk.Button(middle_frame, text='4.1 Global Threshold', font=("Helvetica", 18), command=fun4_1)
b41.grid(row=3, sticky='nsew', padx=20, pady=20)
b42 = tk.Button(middle_frame, text='4.2 Local Threshold', font=("Helvetica", 18), command=fun4_2)
b42.grid(row=4, sticky='nsew', padx=20, pady=20)
b51 = tk.Button(lfpara, text='5.1 Rotation, Scaling, Translation', font=("Helvetica", 18), command=fun5_1)
b51.grid(row=1, sticky='nsew', padx=5, pady=20)
b52 = tk.Button(lf51, text='5.2 Perspective Transform', font=("Helvetica", 18), command=fun5_2)
b52.grid(row=1, sticky='nsew', padx=5, pady=20)

# all entries
e_angle = tk.Entry(lfparas)
e_angle.grid(row=0, column=1, sticky='nsew', pady=10)
e_scale = tk.Entry(lfparas)
e_scale.grid(row=1, column=1, sticky='nsew', pady=10)
e_tx = tk.Entry(lfparas)
e_tx.grid(row=2, column=1, sticky='nsew', pady=10)
e_ty = tk.Entry(lfparas)
e_ty.grid(row=3, column=1, sticky='nsew', pady=10)

main_window.mainloop()
