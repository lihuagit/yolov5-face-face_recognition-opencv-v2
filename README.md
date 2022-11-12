# yolov5-face-face_recognition-opencv-v2
更新的yolov5检测人脸和关键点，只依赖opencv库就可以运行，程序包含C++和Python两个版本的。

python使用face_recognition库做了人脸匹配，可以用做人脸识别，但是识别精度不高

对比上一个版本的，现在这个版本的不同之处在于：

(1).分辨率调整为640x640

(2).提供了yolov5s, yolov5m, yolov5l三种检测人脸+关键点的模型

(3). 后处理方式稍有不同

(4). 在yolov5网络结构里的第一个模块是StemBlock，不再是FCOUS

onnx文件在百度云盘，链接：https://pan.baidu.com/s/1UcMjnAcP5O_I2gW7gUC36Q 
提取码：1234

## py使用方式

下载依赖
```shell
pip install opencv-python face_recognition
```

运行
```shell
python main.py \
        --yolo_type [权重路径] \
        --imgpath   [待预测图片路径] \
        --is_video  [True/False] \
        --videopath [待预测视频路径]
```

例如：
```shell
python main.py --yolo_type yolov5m --imgpath selfie.jpg
python main.py --yolo_type yolov5s --is_video True --videopath cai.flv
```