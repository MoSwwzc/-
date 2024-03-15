import datetime
import time
import tkinter as tk  # 使用Tkinter前需要先导入
from datetime import datetime
from tkinter import *

from PIL import Image

from segment import *

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
        cv2.namedWindow('camera', 1)
        # 摄像头
        video = 'http://admin:admin@192.168.3.179:8081/video'
        capture = cv2.VideoCapture(video)  # 电脑自身的摄像头
        while True:
            success, img = capture.read()
            cv2.imshow("camera", img)
            # 按键处理
            key = cv2.waitKey(10)
            if key == 27:
                # esc
                break
            if key == 32:
                # 空格按键
                filename = 'frame.jpg'
                img = img[:, 0:int(img.shape[1] / 2), :]
                cv2.imwrite(filename, img)
                img = Image.open('frame.jpg')
                img = img.transpose(Image.ROTATE_270)
                img.save("original.jpg")
                break
        capture.release()
        cv2.destroyWindow("camera")

        qrcode = cv2.imread('original.jpg')
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        qrcode = cv2.filter2D(qrcode, -1, kernel=kernel)

        qr_detect = cv2.QRCodeDetector()
        data, bbox, st_qrcode = qr_detect.detectAndDecode(qrcode)
        pts1 = np.float32([bbox[0][0], bbox[0][1],
                           bbox[0][3], bbox[0][2]])
        pts2 = np.float32([[321, 926], [758, 926],
                           [321, 1363], [758, 1363]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(qrcode, matrix, (1100, 2200),borderValue=(255,255,255))
        cv2.imwrite("frame1.jpg", result)
        newpicture = result[1363:]
        cv2.imwrite("frame.jpg", newpicture)
        result=segment_and_pred("frame.jpg","segmentation/")
        pint_1 = result.index('\n')
        pint_2 = result.index('\n', result.index('\n') + 1)
        now_time = datetime.now()
        str_time = result[pint_1 + 1:pint_2 - 1]
        dt_time = datetime.strptime(str_time,'%Y/%m/%d %H:%M:%S')
        pint = result.index(' / EXpired')
        name = result[0:pint - 1]



        if now_time < dt_time:
            var.set("Please pass through")
            print(name + " Please pass through")
            label_img.configure(
                image=img_gif0)
        else:
            label_img.configure(image=img_gif1)
            var.set("Not passable, please try again later!")
            print(name + " Not passable, please try again later!")
        shutil.rmtree("segmentation")
    else:
        on_hit = False
        var.set('')

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