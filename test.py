import tkinter as tk  # 使用Tkinter前需要先导入
import time
from PIL import Image
from PIL import ImageTk
from tkinter import *
from pyzbar.pyzbar import decode
from PIL import Image
from segment import *
import datetime
from datetime import datetime

# 第1步，实例化object，建立窗口window
window = tk.Tk()
# 第2步，给窗口的可视化起名字
window.title('My Window')
# 第3步，设定窗口的大小(长 * 宽)
window.geometry('1000x800')  # 这里的乘是小x


# window.overrideredirect(1)
# def myquit(*args):
#     window.destroy()
# window.bind("<Any-KeyPress>", myquit)
# 第4步，在图形界面上设定标签
var = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
var_1 = tk.StringVar()
welcome = tk.StringVar()
ttime = tk.StringVar()
img_gif = tk.PhotoImage(file='白色.gif')  # 设置图片
img_gif0 = tk.PhotoImage(file='可通行.gif')
img_gif1 = tk.PhotoImage(file='不可通行.gif')
img_gif2 = tk.PhotoImage(file='中国海关.gif')

# label1 = tk.Label(window, bg='white', width=40, height=5)  # create a label to insert this image
# label1.grid()  # set the label in the main window
label_img0 = tk.Label(window, image=img_gif2, bg='DodgerBlue', width=642, height=55, anchor="w")
label_img0.pack(side=TOP)
t = tk.Label(window, textvariable=ttime, bg='white', fg='black', font=('DFKai-SB', 12), width=80, height=2)

w = tk.Label(window, textvariable=var_1, bg='DodgerBlue', fg='white', font=('DIN', 24), width=40, height=2)
w.pack()
var_1.set('Customs Health Declaration')
e = tk.Label(window, textvariable=welcome, bg='DodgerBlue', fg='IndianRed2', font=('Frutiger', 24), width=40, height=2)
e.pack()
welcome.set('Welcome to China Customs!')
label_img = tk.Label(window, image=img_gif, bg='white', width=640, height=300)
label_img.pack()
l = tk.Label(window, textvariable=var, bg='white', fg='black', font=('DFKai-SB', 24), width=40, height=2)
# 说明： bg为背景，fg为字体颜色，font为字体，width为长，height为高，这里的长和高是字符的长和高，比如height=2,就是标签有2个字符这么高
l.pack()
var.set('Please scan the QR code of \n the customs clearance voucher')


def gettime():
    ttime.set(time.strftime("%H:%M:%S"))  # 获取当前时间
    window.after(1000, gettime)

# 定义一个函数功能（内容自己自由编写），供点击Button按键时调用，调用命令参数command=函数名
on_hit = False
def hit_me_1():
    global on_hit
    if on_hit == False:
        on_hit = True
        result = "Yu Hang ENTRY  EXpired at\n2023/6/10 18:18:03\nFOR YOUR CONVENIENCE，IT IS SUGGESTED\nTO CAPTURE AND SAVE THE CODE FOR\nDECLARATION"
        pint = result.index('ENTRY')
        pint_1 = result.index('\n')
        pint_2 = result.index('\n', result.index('\n')+1)
        now_time = datetime.now()
        # print(now_time)
        # print(type(now_time))
        str_time = result[pint_1+1:pint_2-1]
        # rint(str_time)p
        dt_time = datetime.strptime(str_time, '%Y/%m/%d %H:%M:%S')  #代码继承了网络创作者的想法，非常感谢他的帮助https://zhuanlan.zhihu.com/p/337296461
        # print(dt_time)
        # print(type(dt_time))
        name = result[0:pint-1]
        if now_time < dt_time:
            var.set("Please pass through")
            print(name + "Please pass through")
            label_img.configure(image=img_gif0) #Tkinter 转换图像的想法来源于网络创作者https://blog.csdn.net/weixin_39518984/article/details/115480844
        else:
            label_img.configure(image=img_gif1)
            var.set("Not passable, please try again later!")
            print(name + "Not passable, please try again later!")
        # var.set(result)
        # print(result)
    else:
        on_hit = False
        var.set('')

    # cv2.namedWindow('camera', 1)
    # # 摄像头
    # video = 'http://admin:admin@172.28.138.90:8081/video'
    # capture = cv2.VideoCapture(video)  # 电脑自身的摄像头
    # while True:
    #     success, img = capture.read()
    #     cv2.imshow("camera", img)
    #     # 按键处理
    #     key = cv2.waitKey(10)
    #     if key == 27:
    #         # esc
    #         break
    #     if key == 32:
    #         # 空格按键
    #         filename = 'frame.jpg'
    #         img = img[:, 0:int(img.shape[1] / 2), :]
    #         cv2.imwrite(filename, img)
    #         img = Image.open('frame.jpg')
    #         img = img.transpose(Image.ROTATE_270)
    #         img.save("frame.jpg")
    # capture.release()
    # cv2.destroyWindow("camera")
    #
    # qrcode = cv2.imread('frame.jpg')
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    # qrcode = cv2.filter2D(qrcode, -1, kernel=kernel)
    # data = decode(qrcode)
    # (l, t, w, h) = data[0].rect
    # pts1 = np.float32([[l, t], [l + w, t],
    #                    [l, t + h], [l + w, t + h]])
    # pts2 = np.float32([[188, 860], [588, 860],
    #                    [188, 1260], [588, 1260]])
    # matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # result = cv2.warpPerspective(qrcode, matrix, (770, 2200))
    # newpicture = result[1261:2200]
    # newpicture.cv2.imwrite("frame.jpg")
    # result=segment_and_pred("frame.jpg","segmentation/")
def hit_me_2():
    window.destroy()
t.pack()
gettime()

# 第5步，在窗口界面设置放置Button按键
a = tk.Button(window, text='Check', font=('Arial', 12), width=7, height=2, command=hit_me_1)

b = tk.Button(window, text='Quit', font=('Arial', 12), width=7, height=2, command=hit_me_2)
a.place(x=300,y=700)
b.place(x=600,y=700)
# 第6步，主窗口循环显示
window.mainloop()
