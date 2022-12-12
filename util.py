import cv2
import os
import numpy as np


im_path = './support/street'
bbox_path = './support/street_detection2d_results'
line_path = './support/street_line_results'
LiDAR_detection_path = './support/LiDAR_detection3d_results'

im_name_list = os.listdir(im_path)
# print(im_name_list)
im_name_list.sort()
# print(im_name_list)

hide_conf = False
auto_play = True
frame_rate = 20
bbox_color_dict = {'car': (0, 128, 0), 'truck': (128, 0, 0), 'others': (0, 0, 128)}
carline_color_dict = {0: (0, 128, 128), 1: (128, 0, 128), 2: (128, 128, 0), 'others': (128, 128, 128)}


names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


class Annotator:

    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im_idx, font_size=None, font='Arial.ttf', pil=False, example='abc'):

        # use cv2
        self.im_idx = im_idx
        self.im, self.w, self.h = self.read_image()
        # self.box_lw = box_lw or max(round(sum(self.im.shape) / 2 * 0.003), 2)  # line width
        self.box_lw = 2
        self.carline_lw = 5

        self.label_list, self.xyxy_list, self.conf_list = self.read_bboxes()
        self.line_list = self.read_lines()

        self.detected_object_num = len(self.label_list)
        self.detected_carline_num = len(self.line_list)

        # self.draw_bboxes()
        # self.draw_lines()

        self.publish()

    def read_bboxes(self):
        label_list, xyxy_list, conf_list = [], [], []

        txt_name = self.im_idx + '.txt'
        txt_name_full = os.path.join(bbox_path, txt_name)

        with open(txt_name_full) as f:
            rds = f.readlines()
            for rd in rds:
                cls, x, y, w, h, conf = rd.strip().split(' ')
                cls, x, y, w, h, conf = float(cls), float(x), float(y), float(w), float(h), float(conf)
                xyxy = [(x-w/2)*self.w, (y-h/2)*self.h, (x+w/2)*self.w, (y+h/2)*self.h]
                label = names[int(cls)]  # integer class

                label_list.append(label)
                xyxy_list.append([int(i) for i in xyxy])
                conf_list.append(conf)

        return label_list, xyxy_list, conf_list

    def read_lines(self):
        line_list = []

        txt_name = self.im_idx + '.lines.txt'
        txt_name_full = os.path.join(line_path, txt_name)
        with open(txt_name_full) as f:
            rds = f.readlines()

            for rd in rds:
                rd = rd.strip().split()
                x = [int(float(i)) for i in rd[0::2]]
                y = [int(float(i)) for i in rd[1::2]]
                line = []
                for i in range(len(x)):
                    line.append((x[i], y[i]))

                line_list.append(line)
        return line_list

    def read_image(self):
        im_name = self.im_idx + '.jpg'
        im_name_full = os.path.join(im_path, im_name)
        im = cv2.imread(im_name_full)
        h, w, _ = im.shape
        return im, w, h

    def draw_bboxes(self, txt_color=(255, 255, 255)):
        bbox_num = len(self.label_list)
        for i in range(bbox_num):
            label = self.label_list[i]
            box = self.xyxy_list[i]
            conf = self.conf_list[i]

            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

            try:
                color = bbox_color_dict[label]
            except KeyError:
                color = bbox_color_dict['others']

            cv2.rectangle(self.im, p1, p2, color, thickness=self.box_lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.box_lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.box_lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h - 3 >= 0  # label fits outside box
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.box_lw / 3, txt_color,
                            thickness=tf, lineType=cv2.LINE_AA)
        return self.im

    def draw_lines(self):
        line_num = len(self.line_list)

        for i in range(line_num):
            line = self.line_list[i]
            try:
                color = carline_color_dict[i]
            except KeyError:
                color = carline_color_dict['others']
            for i in range(len(line)-1):
                cv2.line(self.im, line[i], line[i+1], color=color, thickness=self.carline_lw)

        return self.im

    def show_res(self):
        if auto_play:
            cv2.namedWindow('frame')
            cv2.imshow('frame', self.im)
            cv2.waitKey(int(1000 / frame_rate))
            cv2.destroyWindow(self.im_idx)

    def calulate_detections(self):
        obj_dict = {}
        for obj in self.label_list:
            try:
                obj_dict[obj] += 1
            except KeyError:
                obj_dict.update({obj: 1})
        # print(obj_dict)
        return obj_dict

    def publish(self):
        # self.im = cv2.cvtColor(self.im, cv2.COLOR_RGB2BGR)
        # self.im = cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB)
        # self.im = self.im[..., ::-1]
        return self.im, self.w, self.h, self.detected_object_num, self.detected_carline_num


if __name__ == '__main__':
    for im_name in im_name_list:
        im_idx = im_name[:-4]
        annotator = Annotator(im_idx=im_idx)
        annotator.show_res()
        annotator.calulate_detections()


