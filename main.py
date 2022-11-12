import os
from pathlib import Path
import cv2
import argparse
import numpy as np
import face_recognition

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm' 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

class yolov5():
    def __init__(self, yolo_type, confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.5):
        anchors = [[4,5,  8,10,  13,16], [23,29,  43,55,  73,105], [146,217,  231,300,  335,433]]
        num_classes = 1
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.no = num_classes + 5 + 10
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        self.inpWidth = 640
        self.inpHeight = 640
        self.net = cv2.dnn.readNet(yolo_type+'-face.onnx')
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        ratioh, ratiow = frameHeight / self.inpHeight, frameWidth / self.inpWidth
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.

        confidences = []
        boxes = []
        landmarks = []
        for detection in outs:
            confidence = detection[15]
            # if confidence > self.confThreshold and detection[4] > self.objThreshold:
            if detection[4] > self.objThreshold:
                center_x = int(detection[0] * ratiow)
                center_y = int(detection[1] * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                landmark = detection[5:15] * np.tile(np.float32([ratiow,ratioh]), 5)
                landmarks.append(landmark.astype(np.int32))
        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            landmark = landmarks[i]
            frame = self.drawPred(frame, confidences[i], left, top, left + width, top + height, landmark)
        return frame
    def drawPred(self, frame, conf, left, top, right, bottom, landmark):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)

        path = ROOT/'tar' #文件夹目录
        files= os.listdir(path)
        fimgs = []
        for file in files: #遍历文件夹
            if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
                source = path/file
                is_file = Path(source).suffix[1:] in (IMG_FORMATS)
                if(is_file == True):
                    fimgs.append(source) #每个文图片存到list中
        # print(fimgs) #打印结果
        imax = 0.
        nmax = ''
        for fimg in fimgs:
            try:
                roi = frame[top:bottom, left:right]
                roi = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
                fileimg=cv2.imread(str(fimg))
                fileimg = cv2.cvtColor(fileimg,cv2.COLOR_BGR2RGB)
                roi_encode=face_recognition.face_encodings(roi)[0]
                fileimg_encode=face_recognition.face_encodings(fileimg)[0]
                res = 1 - face_recognition.face_distance([roi_encode], fileimg_encode)
                filename=os.path.basename(fimg)
                filename=filename.split('.')[0]
            except:
                res=0
            if(res>imax):
                imax=res
                nmax=filename
        if(imax<0.4):
            nmax='null'
        cv2.putText(frame, nmax, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # label = '%.2f' % conf
        # Display the label at the top of the bounding box
        # labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # top = max(top, labelSize[1])
        # cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        for i in range(5):
            cv2.circle(frame, (landmark[i*2], landmark[i*2+1]), 1, (0,255,0), thickness=-1)
        return frame
    def detect(self, srcimg):
        # 在使用神经网络进行图像识别及检测时，需要对数据进行预处理
        # srcimg：原图
        # 1 / 255.0 ：归一化
        # 预处理size：(self.inpWidth, self.inpHeight)
        # [0, 0, 0]三通道的一个减去的值
        # swapRB： BGR转RGB
        # crop:图像裁剪,默认为False.当值为True时，先按比例缩放，然后从中心裁剪成size尺寸
        blob = cv2.dnn.blobFromImage(srcimg, 1 / 255.0, (self.inpWidth, self.inpHeight), [0, 0, 0], swapRB=True, crop=False)
        
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]

        # inference output
        outs[..., [0,1,2,3,4,15]] = 1 / (1 + np.exp(-outs[..., [0,1,2,3,4,15]]))   ###sigmoid
        row_ind = 0
        for i in range(self.nl):
            h, w = int(self.inpHeight/self.stride[i]), int(self.inpWidth/self.stride[i])
            length = int(self.na * h * w)
            if self.grid[i].shape[2:4] != (h,w):
                self.grid[i] = self._make_grid(w, h)
            
            g_i = np.tile(self.grid[i], (self.na, 1))
            a_g_i = np.repeat(self.anchor_grid[i], h * w, axis=0)
            outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + g_i) * int(self.stride[i])
            outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * a_g_i

            outs[row_ind:row_ind + length, 5:7] = outs[row_ind:row_ind + length, 5:7] * a_g_i + g_i * int(self.stride[i])   # landmark x1 y1
            outs[row_ind:row_ind + length, 7:9] = outs[row_ind:row_ind + length, 7:9] * a_g_i + g_i * int(self.stride[i])  # landmark x2 y2
            outs[row_ind:row_ind + length, 9:11] = outs[row_ind:row_ind + length, 9:11] * a_g_i + g_i * int(self.stride[i])  # landmark x3 y3
            outs[row_ind:row_ind + length, 11:13] = outs[row_ind:row_ind + length, 11:13] * a_g_i + g_i * int(self.stride[i])  # landmark x4 y4
            outs[row_ind:row_ind + length, 13:15] = outs[row_ind:row_ind + length, 13:15] * a_g_i + g_i * int(self.stride[i])  # landmark x5 y5
            row_ind += length
        return outs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_type', type=str, default='yolov5m', choices=['yolov5s', 'yolov5m', 'yolov5l'], help="yolo type")
    parser.add_argument("--imgpath", type=str, default='selfie.jpg', help="image path")
    parser.add_argument("--is_video", type=str, default=False, help="image path")
    parser.add_argument("--videopath", type=str, default='cai.flv', help="image path")
    parser.add_argument('--confThreshold', default=0.4, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    parser.add_argument('--objThreshold', default=0.3, type=float, help='object confidence')
    args = parser.parse_args()

    yolonet = yolov5(args.yolo_type, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold, objThreshold=args.objThreshold)
    
    is_video = False
    if(args.is_video == True): is_video = True

    if (is_video):
        video = cv2.VideoCapture()
        video.open(args.videopath)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi',fourcc, 30.0, (640,480))
        while True :
            sucess,img=video.read()
            if(sucess == False):
                break
            img = cv2.resize(img, (640, 480))
            
            dets = yolonet.detect(img)
            srcimg = yolonet.postprocess(img, dets)
            out.write(srcimg)
            winName = 'Deep learning object detection in OpenCV'
            cv2.namedWindow(winName, 0)
            cv2.imshow(winName, srcimg)
            #显示摄像头
            # cv2.imshow("iLucAs",img)
            cv2.waitKey(1)
        video.release()
        out.release()
        cv2.destroyAllWindows()

    srcimg = cv2.imread(args.imgpath)
    dets = yolonet.detect(srcimg)
    srcimg = yolonet.postprocess(srcimg, dets)

    winName = 'Deep learning object detection in OpenCV'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()