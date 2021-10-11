import torch, os, cv2
from model.model import parsingNet
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.constant import culane_row_anchor, tusimple_row_anchor
from scipy.optimize import leastsq
from yolo import YOLO
from PIL import Image

#标定和基础参考信息
#近似针孔成像模型,近似fx=fy
fx = 3127.5753
#马路宽度，以m为单位
road_width = 3.75
#输入视频的尺寸,需要注意的是尺寸不对会导致输出出错
img_w, img_h = 1920, 1080
#内参矩阵和畸变矩阵
mtx =  [[3.12757530e+03, 0.00000000e+00, 1.02648203e+03],
 [0.00000000e+00, 3.12744985e+03, 5.06658577e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
dist =  [[-5.32539241e-02,  1.97503055e+01, -1.36881263e-04,\
  3.43796125e-03, -5.23363192e+02]]

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    #载入车道检测模型
    net = parsingNet(pretrained = False, backbone='18',cls_dim = (201,18,4),
                    use_aux=False).cuda() 
    state_dict = torch.load(r'./culane_18.pth', map_location='cuda:0')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    #标准化
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    #车道线识别网络锚点位置
    row_anchor = culane_row_anchor

    #定义物体检测网络
    yolo = YOLO(image=True)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    #获取视频输入地址
    vin = input("Input video path:")
    vin = cv2.VideoCapture(vin)
    vout = cv2.VideoWriter('./testout.avi', fourcc , 24.0, (img_w, img_h))
    
    #处理帧计数，用于debug
    frame_count = 0

    #开始处理视频流
    while True:
        #读取得到cv2格式帧
        success, origin_img = vin.read()
        #反畸变。由于手机拍摄畸变很小，这里作为附加功能提供
        #origin_img = cv2.undistort(origin_img, mtx, dist, None, newcameramtx)
        if success == False:
            break

        #转换为PIL格式并喂给gpu
        img_PIL = Image.fromarray(cv2.cvtColor(origin_img,cv2.COLOR_BGR2RGB))
        imgs = img_transforms(img_PIL)
        imgs = imgs[None, :, :, :]
        imgs = imgs.cuda()

        col_sample = np.linspace(0, 800 - 1, 200)
        col_sample_w = col_sample[1] - col_sample[0]

        #获取车道线检测信息
        with torch.no_grad():
            out = net(imgs)

        global out_j
        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(200) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == 200] = 0
        out_j = loc

        #检测物体并绘制bounding box
        img = img_PIL
        r_image, frame = yolo.detect_image(img)

        #提取bounding box的底部和中心，用于绘制距离信息
        bottom = []
        center = []
        if frame != []:
            bottom = frame[:, 2]
            center = [(frame[:, 0] + frame[:, 2])/2, (frame[:, 1] + frame[:, 3])/2]

        #转换为openCV格式
        vis = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
          
        #更远的车道线，由于本实验测距仅使用中间车道线，此处作为附加功能注释
        '''
        for ii in range(out_j.shape[1]):
            if np.sum(out_j[:, ii] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, ii] > 0:
                        ppp = (int(out_j[k, ii] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[18-1-k]/288)) - 1)
                        cv2.circle(vis,ppp,5,(0,255,0),-1)
        '''

        out_j = out_j[:, 1:3]

        #读取有效数据范围end，不大于锚点数
        end = 0
        for i in range(18):
            if(sum(out_j[i]) != 0):
                end = i 

        #使用最小二乘法拟合车道线
        #翻转row_anchor
        in_row_anchor = row_anchor[:]
        in_row_anchor.reverse() 

        #反函数，用于根据纵坐标求横坐标
        def inFunc(p, y):
            k, b = p
            return (y-b)/k

        #最小二乘法
        def OLS(Xi, Yi):    
            def func(p, x):
                k, b = p
                return k*x+b

            def error(p, x, y):
                return (func(p, x)-y)**2

            p0 = [1, 10]
            #print((Xi, Yi))
            Para=leastsq(error,p0,args=(Xi,Yi))
            return Para[0]

        #bbox与车道交点坐标
        bottom_p1 = []
        bottom_p2 = []

        #分别绘制两条车道线的拟合直线
        if(out_j[0][0] > 0 and out_j[end][0]>0 and end>=2):
            Xi = out_j[0:end+1, 0]
            Yi = in_row_anchor[0:end+1]
            p = OLS(Xi, Yi)
            Y0 = Yi[0]
            X0 = inFunc(p, Y0)
            Y1 = Yi[end]
            X1 = inFunc(p, Y1)
            cv2.line(vis, tuple([int(X0/800*img_w* col_sample_w), int(Y0/288*img_h)-1]), tuple([int(X1/800*img_w* col_sample_w), int(Y1/288*img_h)-1]), (0,0,255),10,8)
            for b in bottom:
                bottom_p1.append(inFunc(p, b))

        if(out_j[0][1] > 0 and out_j[end][1]>0 and end>=2):
            Xi = out_j[0:end+1, 1]
            Yi = in_row_anchor[0:end+1]
            p = OLS(Xi, Yi)
            Y2 = Yi[0]
            X2 = inFunc(p, Y2)
            Y3 = Yi[end]
            X3 = inFunc(p, Y3)
            cv2.line(vis, tuple([int(X2/800*img_w* col_sample_w), int(Y2/288*img_h)-1]), tuple([int(X3/800*img_w* col_sample_w), int(Y3/288*img_h)-1]), (0,0,255),10,8)
            for b in bottom:
                bottom_p2.append(inFunc(p, b))

        #距离估计
        if bottom_p1!=[] and bottom_p2!=[]:
            for i in range(len(bottom_p1)):
                pixel_len = abs(bottom_p1[i]-bottom_p2[i])
                distance = road_width*fx*(1/pixel_len)
                cv2.putText(vis,str(round(distance, 1)) + 'm', (int(center[1][i])-80, int(center[0][i])), cv2.FONT_HERSHEY_SIMPLEX, 1.2,(255,255,255), 1, cv2.LINE_AA)

        vout.write(vis)
        print("frame: ", frame_count)
        frame_count += 1
    
    vin.release()
    vout.release()