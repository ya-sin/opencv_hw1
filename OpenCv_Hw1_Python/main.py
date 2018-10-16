import cv2
import numpy as np
import tkinter as tk
import PIL.Image, PIL.ImageTk

def fun1_1():
  dog = cv2.imread("../img/dog.bmp", cv2.IMREAD_COLOR)
  print('Height:', dog.shape[0])
  print('width:', dog.shape[1])
  cv2.imshow('TT',dog)

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
  # fig = gcf()
  # fig.canvas.manager.window.raise_()
  dst = cv2.addWeighted(img1,0,img2,1,0) #dst = img*1+dst*0+gamma
  cv2.imshow('Blending',dst)
def fun2_1():
  screw = cv2.imread("../img/M8.jpg", cv2.IMREAD_COLOR)
  grayImg = cv2.cvtColor(screw, cv2.COLOR_BGR2GRAY)
  blurImg = cv2.GaussianBlur(grayImg,(3,3),0)
  cv2.imshow('Gray and Smooth',blurImg)

  kernelX = [
        [ -1, 0, 1 ],
        [ -2, 0, 2 ],
        [ -1, 0, 1 ]
    ]

  kernelY = [
      [ -1, -2, -1 ],
      [  0,  0,  0 ],
      [  1,  2,  1 ]
  ]

  kernelXMat = np.array(kernelX, dtype=float)
  kernelYMat = np.array(kernelY, dtype=float)

  horizGradient = cv2.filter2D(blurImg, cv2.CV_32F, kernelXMat)
  vertGradient = cv2.filter2D(blurImg, cv2.CV_32F, kernelYMat)

  abs_horizGradient = cv2.convertScaleAbs(horizGradient)
  abs_vertGradient = cv2.convertScaleAbs(vertGradient)

  ret, result_H = cv2.threshold(abs_horizGradient, 40, 255, cv2.THRESH_BINARY)
  ret, result_V = cv2.threshold(abs_vertGradient, 40, 255, cv2.THRESH_BINARY)

  cv2.imshow('horizontal edge',result_H)
  cv2.imshow('vertical edge',result_V)

  def Change(x):
      thre = cv2.getTrackbarPos('magnitude', 'Magnitude')
      dst = cv2.addWeighted(abs_horizGradient,0.5,abs_vertGradient,0.5,0)
      ret, dst = cv2.threshold(dst, thre, 255, cv2.THRESH_BINARY)
      cv2.imshow('Magnitude',dst)
  cv2.namedWindow('Magnitude')
  cv2.createTrackbar('magnitude', 'Magnitude', 40,255,Change)
  dst = cv2.addWeighted(abs_horizGradient,0.5,abs_vertGradient,0.5,0)
  ret, dst = cv2.threshold(dst, 40, 255, cv2.THRESH_BINARY)
  cv2.imshow('Magnitude',dst)
def fun3_1():
  pyramids = cv2.imread("../img/pyramids_Gray.jpg", cv2.IMREAD_COLOR)
  GauPyra_1 = cv2.pyrDown(pyramids)
  LapPyra_0 = pyramids - cv2.pyrUp(GauPyra_1)
  GauPyra_2 = cv2.pyrDown(GauPyra_1)
  LapPyra_1 = GauPyra_1 - cv2.pyrUp(GauPyra_2)
  InvPyra_0 = LapPyra_0 + cv2.pyrUp(GauPyra_1)
  # cv2.imshow('origin', pyramids)
  # cv2.imshow('Laplacian pyramid level 1 ', LapPyra_1)
  cv2.imshow('Gaussian pyramid level 1', GauPyra_1)
  cv2.imshow('Laplacian pyramid level 0 ', LapPyra_0)
  cv2.imshow('Inverse pyramid level 1 ', GauPyra_1)
  cv2.imshow('Inverse pyramid level 0 ', InvPyra_0)
def fun4_1():
  QR = cv2.imread("../img/QR.png", cv2.IMREAD_COLOR)
  grayImg = cv2.cvtColor(QR, cv2.COLOR_BGR2GRAY)
  ret, result = cv2.threshold(grayImg,80,255,cv2.THRESH_BINARY)
  cv2.imshow('Original', QR)
  cv2.imshow('Threshold', result)
def fun4_2():
  QR = cv2.imread("../img/QR.png", cv2.IMREAD_COLOR)
  grayImg = cv2.cvtColor(QR, cv2.COLOR_BGR2GRAY)
  grayImg = cv2.medianBlur(grayImg,5)
  result = cv2.adaptiveThreshold(grayImg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,19,-1)
  cv2.imshow('Original', QR)
  cv2.imshow('Adapted Threshold', result)
def fun5_1():
  # edtAngle, edtScale. edtTx, edtTy to access to the ui object
  Angle = float(e_angle.get())
  Scale = float(e_scale.get())
  Tx = float(e_tx.get())
  Ty = float(e_ty.get())
  img = cv2.imread('../img/OriginalTransform.png')
  H = np.float32([[1,0,Tx],[0,1,Ty]])
  rows,cols = img.shape[:2]
  res = cv2.warpAffine(img,H,(rows,cols))
  # rotate & Scale]
  rows,cols = res.shape[:2]
  M = cv2.getRotationMatrix2D((130+Tx,125+Ty),Angle,Scale)
  res = cv2.warpAffine(res,M,(rows,cols))
  cv2.imshow(', Scaling, Translation',img)
  cv2.imshow('Rotation, Scaling, Translation',res)
def fun5_2():
  imgpoints=[]
  objpoints = [[20,20],[450,20],[450,450],[20,450]]
  def draw_circle(event,x,y,flags,param):
      if event == cv2.EVENT_LBUTTONDOWN:
          nonlocal imgpoints
          imgpoints.append([x,y])
          if(len(imgpoints) <= 4):
              cv2.circle(img,(x,y), 10, (0,0,255), -1)
              cv2.imshow('Original',img)

          if len(imgpoints)==4:
              pts1 = np.float32(imgpoints)
              pts2 = np.float32(objpoints)
              M = cv2.getPerspectiveTransform(pts1,pts2)
              dst = cv2.warpPerspective(perspect,M,(450,450))

              cv2.imshow('Perspective image', dst)

  img = cv2.imread('../img/OriginalPerspective.png')
  perspect = img.copy()
  cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
  cv2.imshow('Original',img)
  cv2.setMouseCallback('Original',draw_circle)


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
