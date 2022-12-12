"""
Description: RRML ADS PLATFORM v1.0
Funcs:
    - Five Buttons: 1. Original Video
                    2. Object Detection
                    3. Car Line Detection
                    4. Fusion (Object Detection + Car Line Detection)
                    5. LiDAR Detection
    - One QLabel shower
    - One textEditor

update: 7/Dec/2022
Created by Lin

"""
import os
import cv2

from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QPlainTextEdit, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile, QTimer
from PySide2.QtGui import QPixmap, QImage
from PySide2.QtGui import QIcon


from util import Annotator
from util import im_path, bbox_path, line_path, LiDAR_detection_path

FRAME_RATE = 20

class Viewer:

    def __init__(self):

        # load ui
        ui_file = QFile('ui/ui.ui')
        ui_file.open(QFile.ReadOnly)
        ui_file.close()
        self.ui = QUiLoader().load(ui_file)

        self.bnx_clicked = None
        # handle event
        self.btn0_event()
        self.btn1_event()
        self.btn2_event()
        self.btn3_event()
        self.btn4_event()


        # define value
        self.timer0 = QTimer()
        self.timer1 = QTimer()
        self.timer2 = QTimer()
        self.timer3 = QTimer()
        self.timer4 = QTimer()

        # 2d
        self.im_path = im_path
        self.im_name_list = os.listdir(self.im_path)
        self.im_name_list.sort()
        self.idx = 0

        # 3d
        self.im_3d_path = LiDAR_detection_path  # pcd image path
        self.im_3d_name_list = os.listdir(self.im_3d_path)
        self.im_3d_name_list.sort()
        self.idx_3d = 0
        self.obj_num, self.cl_num, self.obj_dict = 0, 0, {}

        # init textedit
        self.ui.textEdit0.setPlainText("waiting for inference")

    def btn0_event(self):
        self.ui.bn0.clicked.connect(self.update0)

    def btn1_event(self):
        self.ui.bn1.clicked.connect(self.update1)

    def btn2_event(self):
        self.ui.bn2.clicked.connect(self.update2)

    def btn3_event(self):
        self.ui.bn3.clicked.connect(self.update3)

    def btn4_event(self):
        self.ui.bn4.clicked.connect(self.update4)

    def update0(self):
        self.bnx_clicked = 0
        # self.timer0.stop()
        self.timer0.start(1000/FRAME_RATE)
        self.timer0.timeout.connect(self.show_image)
        self.timer0.timeout.connect(self.detection_vis)

    def update1(self):
        self.bnx_clicked = 1
        # self.timer1.stop()
        self.timer1.start(1000/FRAME_RATE)
        self.timer1.timeout.connect(self.show_image)

    def update2(self):
        self.bnx_clicked = 2
        # self.timer2.stop()
        self.timer2.start(1000/FRAME_RATE)
        self.timer2.timeout.connect(self.show_image)

    def update3(self):
        self.bnx_clicked = 3
        # self.timer3.stop()
        self.timer3.start(1000/FRAME_RATE)
        self.timer3.timeout.connect(self.show_image)

    def update4(self):
        self.bnx_clicked = 4
        # self.timer3.stop()
        self.timer3.start(1000*2/(FRAME_RATE))
        self.timer3.timeout.connect(self.show_image)

    def show_image(self):

        if self.idx == len(self.im_name_list)-1:
            self.timer0 = QTimer()
            self.timer1 = QTimer()
            self.timer2 = QTimer()
            self.timer3 = QTimer()
            self.timer4 = QTimer()
            self.idx = 0
        if self.idx_3d == len(self.im_3d_name_list)-1:
            self.timer0 = QTimer()
            self.timer1 = QTimer()
            self.timer2 = QTimer()
            self.timer3 = QTimer()
            self.timer4 = QTimer()
            self.idx_3d = 0

        im_idx = self.im_name_list[self.idx][:-4]
        annotator = Annotator(im_idx=im_idx)

        if self.bnx_clicked == 0:   # original video
            im, w, h, _, _ = annotator.publish()
            self.obj_num = 0
            self.cl_num = 0
            self.obj_dict = {}

        elif self.bnx_clicked == 1:  # detection only
            annotator.draw_bboxes()
            im, w, h, self.obj_num, _ = annotator.publish()
            self.obj_dict = annotator.calulate_detections()
            self.cl_num = 0

        elif self.bnx_clicked == 2:     # carline only
            annotator.draw_lines()
            im, w, h, _, self.cl_num = annotator.publish()
            self.obj_num = 0
            self.obj_dict = {}

        elif self.bnx_clicked == 3:     # detection + carline
            annotator.draw_bboxes()
            annotator.draw_lines()
            im, w, h, self.obj_num, self.cl_num = annotator.publish()
            obj_dict = annotator.calulate_detections()

        elif self.bnx_clicked == 4:
            im_3d_name_full = os.path.join(self.im_3d_path, self.im_3d_name_list[self.idx_3d])
            im = cv2.imread(im_3d_name_full)
            h, w, _ = im.shape
            self.idx_3d += 1

        else:
            print('wrong mode')

        if self.bnx_clicked != 4:
            self.detection_vis()
            self.idx += 1
        else:
            self.ui.textEdit0.setPlainText('-------------------LiDAR Detection\n'
                                           'coming soon!!\n')

        im_q = QImage(im, w, h, QImage.Format_BGR888)
        self.ui.shower.setPixmap(QPixmap.fromImage(im_q))
        self.ui.shower.setScaledContents(True)  # 让图片自适应 label 大小

    def detection_vis(self):
        if self.obj_num == 0:
            self.ui.textEdit0.setPlainText('-------------------Object Detection\n'
                                           'objects: {}\n\n'
                                           '-------------------Carline Detection\n'
                                           'car line: {}\n'.format(self.obj_num, self.cl_num))
        else:
            self.ui.textEdit0.setPlainText('-------------------Object Detection\n'
                                           'objects: {} {}\n\n'
                                           '-------------------Carline Detection\n'
                                           'car line: {}\n'.format(self.obj_num, self.obj_dict, self.cl_num))

    # def show_pcd(self):
    #     path = '/home/car/exps/OpenPCDet/tools/jpg'  # image path
    #
    #     im_q = QImage(im, w, h, QImage.Format_BGR888)
    #     self.ui.shower.setPixmap(QPixmap.fromImage(im_q))
    #     self.ui.shower.setScaledContents(True)  # 让图片自适应 label 大小
    #     self.idx += 1


if __name__ == '__main__':
    app = QApplication([])
    v = Viewer()
    v.ui.show()

    app.setWindowIcon(QIcon('./icon/university.png'))
    app.exec_()