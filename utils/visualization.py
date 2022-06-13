import matplotlib.pylab as plt
import torchvision
import os
import torch
import cv2
import numpy as np
from utils.imgname import read_img_name
import matplotlib.pyplot as plt
import seaborn as sns

def featuremap_visual(feature,
                      out_dir='./utils/visualization',  
                      save_feature=True,  
                      show_feature=True,  
                      feature_title=None,  
                      channel = None,
                      num_ch=-1,  
                      nrow=1,  
                      padding=10,  
                      pad_value=1  
                      ):

    # feature = feature.detach().cpu()
    b, c, h, w = feature.shape
    feature = feature[0]
    #'''
    feature_c = feature.view(c, h*w)
    fmax = torch.max(feature_c, dim=-1)[0]
    fmin = torch.min(feature_c, dim=-1)[0]
    feature = (feature - fmin[:, None, None]) / (fmax[:, None, None] - fmin[:, None, None])
    #'''

    feature = feature.unsqueeze(1)
    if channel:
        feature = feature[channel]
    else:
        if c > num_ch > 0:
            feature = feature[:num_ch]

    img = torchvision.utils.make_grid(feature, nrow=nrow, padding=padding, pad_value=pad_value)
    img = img.detach().cpu()
    img = img.numpy()
    images = img.transpose((1, 2, 0))

    # title = str(images.shape) if feature_title is None else str(feature_title)
    title = str('hwc-') + str(h) + '-' + str(w) + '-' + str(c) if feature_title is None else str(feature_title)
    title = title + "_" + str(h) + '-' + str(w)

    plt.title(title)
    plt.imshow(images)
    if save_feature:
        # root=r'C:\Users\Administrator\Desktop\CODE_TJ\123'
        # plt.savefig(os.path.join(root,'1.jpg'))
        out_root = title + '.jpg' if out_dir == '' or out_dir is None else os.path.join(out_dir, title + '.jpg')
        plt.savefig(out_root)
    if show_feature:        plt.show()
    a = 1

def attentionmap_visual(features,
                      out_dir='./utils/visualization',  # 特征图保存路径文件
                      save_feature=True,  # 是否以图片形式保存特征图
                      show_feature=True,  # 是否使用plt显示特征图
                      feature_title=None,  # 特征图名字，默认以shape作为title
                      channel = None,
                      ):

    # feature = feature.detach().cpu()
    b, c, h, w = features.shape
    if b > 6:
        b = 6
    for i in range(b):
        figure = np.zeros(((h+30)*2, (w+30)*(c//2)+30), dtype=np.uint8) + 255
        for j in range(c):
            featureij = features[i, j, :, :]
            fmax = torch.max(featureij)
            fmin = torch.min(featureij)
            #featureij = ((featureij - fmin)/(fmax-fmin))*255
            featureij = (featureij / fmax) * 255
            featureij = featureij.cpu().detach().numpy()
            featureij = featureij.astype('uint8')
            #cv2.imshow("attention-" + str(c), featureij)
            #cv2.waitKey(0)
            if j < c//2:
                figure[10:h+10, 10+(w+20)*j: 10+(w+20)*j+w] = featureij
            else:
                figure[30+h:30+h+h, 10 + (w + 20) * (j-c//2): 10 + (w + 20) * (j-c//2) + w] = featureij
        if feature_title:
            cv2.imwrite(out_dir + '/' + 'attention_' + feature_title + '.png', figure)
        else:
            cv2.imwrite(out_dir + '/' + 'attention_' + str(i) + '_' + str(h) + '.png', figure)
        cv2.imshow("attention-" + str(c), figure)
        cv2.waitKey(0)


def featuremap1d_visual(feature,
                      out_dir='./utils/visualization', 
                      save_feature=True, 
                      show_feature=True,  
                      feature_title=None,  
                      channel = None,
                      num_ch=-1,  
                      nrow=1,  
                      padding=10,  
                      pad_value=1  
                      ):

    # feature = feature.detach().cpu()
    b, c, h, w = feature.shape
    feature = feature[0]
    feature = feature.unsqueeze(1)
    if channel:
        feature = feature[channel]
    else:
        if c > num_ch > 0:
            feature = feature[:num_ch]

    img = torchvision.utils.make_grid(feature, nrow=nrow, padding=padding, pad_value=pad_value)
    img = img.detach().cpu()
    img = img.numpy()
    images = img.transpose((1, 2, 0))

    # title = str(images.shape) if feature_title is None else str(feature_title)
    title = str('hwc-') + str(h) + '-' + str(w) + '-' + str(c) if feature_title is None else str(feature_title)
    title = title + "_" + str(h) + '-' + str(w)

    plt.title(title)
    plt.imshow(images)
    if save_feature:
        # root=r'C:\Users\Administrator\Desktop\CODE_TJ\123'
        # plt.savefig(os.path.join(root,'1.jpg'))
        out_root = title + '.jpg' if out_dir == '' or out_dir is None else os.path.join(out_dir, title + '.jpg')
        plt.savefig(out_root)
    if show_feature:        plt.show()

def visual_box(bbox, name='layer2', scale=1):
    bbox = np.array(bbox.cpu())  # x y x y
    bbox = bbox * scale
    imgpath = read_img_name()
    img = cv2.imread(imgpath)
    bboxnum = bbox.shape[0]
    if bbox.any():
        for i in range(bboxnum):
            cv2.rectangle(img, (int(bbox[i, 0]), int(bbox[i, 1])), (int(bbox[i, 2]), int(bbox[i, 3])), (0, 0, 255), 1)  # x1y1,x2y2,BGR
    filename = os.path.basename(imgpath)
    filename = filename.split('.')[0]
    save_path = './Visualization/localbox/' + name + '/'
    cv2.imwrite(os.path.join(save_path, filename + '.png'), img)

global layer
layer = 0

def attentionheatmap_visual(features,
                      out_dir='./VisualizationP/heatmap/',  
                      save_feature=True,  
                      show_feature=True,  
                      feature_title=None,  
                      channel = None,
                      ):

    # feature = feature.detach().cpu()
    global layer
    b, c, h, w = features.shape
    if b > 1:
        b = 1
    for i in range(b):
        figure = np.zeros(((h+30)*2, (w+30)*(c//2)+30), dtype=np.uint8) + 255
        for j in range(c):
            featureij = features[i, j, :, :]
            featureij = featureij.cpu().detach().numpy()
            #featureij = featureij.astype('uint8')
            fig = sns.heatmap(featureij, cmap="YlGnBu", vmin=0.0, vmax=0.005)  #Wistia, YlGnBu
            fig.set_xticks(range(0))
            #fig.set_xticklabels(f'{c:.1f}' for c in np.arange(0.1, 1.01, 0.1))
            fig.set_yticks(range(0))
            #fig.set_yticklabels(f'{c:.1f}' for c in np.arange(0.1, 1.01, 0.1))
            #sns.despine()
            plt.show()
            plt.close()
            fig_heatmap = fig.get_figure()
            imgpath = read_img_name()
            filename = os.path.basename(imgpath)
            filename = filename.split('.')[0]
            fig_heatmap.savefig(os.path.join(out_dir, filename + '_l' + str(layer) + '_' + str(j) + '.png'))
            layer = (layer + 1) % 4

