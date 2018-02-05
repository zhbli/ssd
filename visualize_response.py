import os
import cv2
import torch
import numpy as np
import tkinter as tk
import torch.backends.cudnn as cudnn
from PIL import Image, ImageTk
from torch.autograd import Variable
from ssd import build_ssd
from data import AnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ ['CUDA_LAUNCH_BLOCKING'] = '1'
feature_size = [38, 19, 10, 5, 3, 1]
feature_size_cumsum = np.array([0, 38*38*4, 19*19*6, 10*10*6, 5*5*6, 3*3*4, 1*1*4]).cumsum()

# VOC_CLASSES = (  # always index 0
#     '1 aeroplane', 'bicycle', 'bird', 'boat',
#     '5 bottle', 'bus', 'car', 'cat', 'chair',
#     '10 cow', 'diningtable', 'dog', 'horse',
#     '14 motorbike', 'person', 'pottedplant',
#     '17 sheep', 'sofa', 'train', 'tvmonitor')

def click(event):
    img_copy = img.copy()
    layer_num = int(event.widget.widgetName)
    row_len = event.y
    col_len = event.x
    rectangle_row_num = row_len // rectangle_size
    rectangle_col_num = col_len // rectangle_size
    idx = feature_size_cumsum[layer_num] + (rectangle_row_num*feature_size[layer_num]+rectangle_col_num)*anchor_num[layer_num] + response_anchor_idx[layer_num][rectangle_row_num][rectangle_col_num] # convert to 1_d index in [0, 8732]
    box = coords[idx]  # coords: ndarray, [8732, 4]
    cv2.rectangle(img_copy, (box[0],box[1]),(box[2],box[3]),(255,0,0),2)
    img_temp = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    img_temp = Image.fromarray(img_temp)
    img_temp = ImageTk.PhotoImage(img_temp)
    panelA.configure(image=img_temp)
    panelA.image = img_temp

def next_img(event):
    img_id = input('img_id: ')
    cls_id = int(input('cls_id: '))
    main(img_id, cls_id)

def main(img_id, cls_idx):
    # load img
    global img, response_anchor_idx, anchor_num
    img_file = '/data/zhbli/VOCdevkit/VOC2007/JPEGImages/{}.jpg'.format(img_id)
    img = cv2.imread(img_file)  # ndarray
    #

    # test img
    transform = BaseTransform(net.size, (104, 117, 123))
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0)).cuda()
    cls_conf_maps, box_deltas = net(x)  # cls_conf_maps: list; boxes_deltas: Tensor, [8732, 4]
    #

    # do softmax
    whole_response = np.zeros([0])
    value_num = np.zeros([1])
    cls_conf_maps_softmax = []
    softmax = torch.nn.Softmax()
    for i in range(len(cls_conf_maps)):
        whole_response = np.append(whole_response, cls_conf_maps[i])
        value_num = np.append(value_num, cls_conf_maps[i].size)
    value_num = value_num.cumsum().astype(np.int)
    whole_response_softmax = softmax(torch.from_numpy(whole_response).view(-1, num_classes)).data.numpy()  # ndarray, [8732, 21]
    whole_response_softmax = whole_response_softmax.reshape(-1)
    for i in range(len(cls_conf_maps)):
        cls_conf_maps_softmax.append(whole_response_softmax[value_num[i]:value_num[i+1]].reshape(cls_conf_maps[i].shape))
    #

    # get cls_idx's response of at every unit
    response = []
    response_anchor_idx = []
    anchor_num = [4, 6, 6, 6, 4, 4]
    for i in range(len(cls_conf_maps_softmax)):
        current_conf_map = cls_conf_maps_softmax[i]
        temp_map1 = current_conf_map[:, :, np.arange(anchor_num[i]) * num_classes + cls_idx]
        temp_map2 = (temp_map1.max(2) * 255).astype(np.int)  # ndarray [height, width]
        response_anchor_idx.append(temp_map1.argmax(2))
        response.append(temp_map2)
    #

    # get boxes' coordinates
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])
    global coords
    coords = (box_deltas.cpu() * scale).cpu().numpy()  # ndarray, [8732, 4]
    #

    # GUI
    global root, panelA, w, rectangle_size
    rectangle_size = 10
    img_temp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_temp = Image.fromarray(img_temp)
    img_temp = ImageTk.PhotoImage(img_temp)
    if panelA is None:
        panelA = tk.Label(root, image=img_temp)
        panelA.image = img_temp
        panelA.pack(side='left')
        panelA.bind('<Button-1>', next_img)
    else:
        panelA.configure(image=img_temp)
        panelA.image = img_temp
        panelA.pack(side='left')
    for i in range(len(response)):
        feature_width = response[i].shape[1]
        feature_height = response[i].shape[0]
        if w[i] is not None:
            w[i].destroy()
        w[i] = tk.Canvas(root, width=rectangle_size * feature_width, height=rectangle_size * feature_height)
        w[i].widgetName = str(i)
        w[i].pack()
        w[i].bind('<Button-1>', click)
        for x in range(feature_height):
            for y in range(feature_width):
                value_hex = hex(response[i][x][y])[2:]
                w[i].create_rectangle(y * rectangle_size, x * rectangle_size, y * rectangle_size + rectangle_size,
                                   x * rectangle_size + rectangle_size, fill='#' + value_hex*3)
    #

if __name__ == '__main__':
    # load net
    num_classes = len(VOC_CLASSES) + 1  # +1 background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load('weights/ssd300_mAP_77.43_v2.pth'))
    net.eval()
    net = net.cuda()
    cudnn.benchmark = True
    print('Finished loading model!')
    #

    # display GUI
    root = tk.Tk()
    panelA = None
    w = [None, None, None, None, None, None]
    img_id = '004932'
    cls_id = 5
    main(img_id, cls_id)
    root.mainloop()
    #
