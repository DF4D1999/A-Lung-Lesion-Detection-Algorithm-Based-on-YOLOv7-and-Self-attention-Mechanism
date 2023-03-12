# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 19:58:16 2022

@author: johnsonlok
"""

import pandas as pd
import os
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from path import Path
from collections import Counter
from ensemble_boxes import weighted_boxes_fusion
import random
import datetime
import os.path as osp
import shutil
import json

def plot_img(img, size=(18, 18), is_rgb=True, title="", cmap='gray'):
    plt.figure(figsize=size)
    plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    plt.show()

def plot_imgs(imgs, cols=2, size=10, is_rgb=True, title="", cmap='gray', img_size=None):
    rows = len(imgs)//cols + 1
    fig = plt.figure(figsize=(cols*size, rows*size))
    for i, img in enumerate(imgs):
        if img_size is not None:
            img = cv2.resize(img, img_size)
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
def draw_bbox(image, box, label, color):  
    alpha = 0.3
    alpha_box = 0.4
    overlay_bbox = image.copy()
    overlay_text = image.copy()
    output = image.copy()

    text_width, text_height = cv2.getTextSize(label.upper(), cv2.FONT_HERSHEY_SIMPLEX, 3, 1)[0]
    cv2.rectangle(overlay_bbox, (box[0], box[1]), (box[2], box[3]),color, -1)
    cv2.addWeighted(overlay_bbox, alpha, output, 1 - alpha, 0, output)
    cv2.rectangle(overlay_text, (box[0], box[1]-7-text_height), (box[0]+text_width+2, box[1]),(0, 0, 0), -1)
    cv2.addWeighted(overlay_text, alpha_box, output, 1 - alpha_box, 0, output)
    cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]),color, thickness)
    cv2.putText(output, label.upper(), (box[0], box[1]-5),cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 1, cv2.LINE_AA)
    
    return output

data = pd.read_csv('train.csv')
print(data.head(), '\nValidating data and images...')

for findfilename in tqdm(data['image_id'].unique()):
    if not (os.path.exists('train/'+findfilename+'.jpg')):
        print("\033[0;31;40m", findfilename, '.png is missing', "\033[0m")
        #print(findfilename,'.png is missing')
        
print('\ntotal anno count: ',len(data))
print('total image count:',len(data['image_id'].unique()))
print(data['class_name'].value_counts())

#======================detele Other lesion=============================================??

#======================Pick out No finding=============================================??
#data.drop(data['class_id']=='No finding')

data.drop(data[data['class_name'] == 'No finding'].index, inplace = True)
data = data.reset_index(drop=True)
data['image_path'] = data['image_id'].map(lambda x:os.path.join('./train', str(x)+'.jpg'))

#======================YOLO data format: label_index, midx, midy, w, h=================
'''data.insert(data.shape[1], 'midx', (data['x_max']+data['x_min'])/2/data['width'])#format()小数位数？
data.insert(data.shape[1], 'midy', (data['y_max']+data['y_min'])/2/data['height'])
data.insert(data.shape[1], 'w', (data['x_max']-data['x_min'])/data['width'])
data.insert(data.shape[1], 'h', (data['y_max']-data['y_min'])/data['height'])'''

#==Visualize Original Bboxes==
labels =  ["Aortic_enlargement", "Atelectasis",        "Calcification", "Cardiomegaly",    "Consolidation",
            "ILD",               "Infiltration",       "Lung_Opacity",  "Nodule/Mass",     "Other_lesion",
            "Pleural_effusion",  "Pleural_thickening", "Pneumothorax",  "Pulmonary_fibrosis"]
viz_labels = labels
label2color = [[59, 238, 119], [222, 21, 229], [94, 49, 164], [206, 221, 133], [117, 75, 3],
                 [210, 224, 119], [211, 176, 166], [63, 7, 197], [102, 65, 77], [194, 134, 175],
                 [209, 219, 50], [255, 44, 47], [89, 125, 149], [110, 27, 100]]
thickness = 3
imgs = []

for img_id, path in zip(data['image_id'][:6], data['image_path'][:6]):

    boxes = data.loc[data['image_id'] == img_id, ['x_min', 'y_min', 'x_max', 'y_max']].values
    img_labels = data.loc[data['image_id'] == img_id, ['class_id']].values.squeeze()
    img = cv2.imread(path)
    
    for label_id, box in zip(img_labels, boxes):
        color = label2color[label_id]
        img = draw_bbox(img, list(np.int_(box)), viz_labels[label_id], color)
    imgs.append(img)

plot_imgs(imgs, size=9, cmap=None)
plt.show()

#================================= BB Fusing Part========================//Date: 230107
iou_thr = 0.5
skip_box_thr = 0.0001
viz_images = []
imagepaths = data['image_path'].unique()

for i, path in tqdm(enumerate(imagepaths[5:8])):
    img_array  = cv2.imread(path)
    image_basename = Path(path).stem
    print(f"(\'{image_basename}\', \'{path}\')")
    img_annotations = data[data.image_id==image_basename]

    boxes_viz = img_annotations[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().tolist()
    labels_viz = img_annotations['class_id'].to_numpy().tolist()
    
    print("Bboxes before nms:\n", boxes_viz)
    print("Labels before nms:\n", labels_viz)
    
    ## Visualize Original Bboxes
    img_before = img_array.copy()
    for box, label in zip(boxes_viz, labels_viz):
        x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])
        color = label2color[int(label)]
        img_before = draw_bbox(img_before, list(np.int_(box)), viz_labels[label], color)
    viz_images.append(img_before)
    
    boxes_list = []
    scores_list = []
    labels_list = []
    weights = []
    
    boxes_single = []
    labels_single = []
    
    cls_ids = img_annotations['class_id'].unique().tolist()
    count_dict = Counter(data['class_id'].tolist())
    print(count_dict)

    for cid in cls_ids:       
        ## Performing Fusing operation only for multiple bboxes with the same label
        if count_dict[cid]==1:
            labels_single.append(cid)
            boxes_single.append(img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().squeeze().tolist())

        else:
            cls_list =img_annotations[img_annotations.class_id==cid]['class_id'].tolist()
            labels_list.append(cls_list)
            bbox = img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
            ## Normalizing Bbox by Image Width and Height
            bbox = bbox/(img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0])
            bbox = np.clip(bbox, 0, 1)
            boxes_list.append(bbox.tolist())
            scores_list.append(np.ones(len(cls_list)).tolist())

            weights.append(1)
            
    # Perform NMS
    boxes, scores, box_labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)#Bboxes fusing method
    
    boxes = boxes*(img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0])
    boxes = boxes.round(1).tolist()
    box_labels = box_labels.astype(int).tolist()

    boxes.extend(boxes_single)
    box_labels.extend(labels_single)
    
    print("Bboxes after nms:\n", boxes)
    print("Labels after nms:\n", box_labels)
    
    ## Visualize Bboxes after operation
    img_after = img_array.copy()
    for box, label in zip(boxes, box_labels):
        color = label2color[int(label)]
        img_after = draw_bbox(img_after, list(np.int_(box)), viz_labels[label], color)
    viz_images.append(img_after)
    print()
        
plot_imgs(viz_images, cmap=None)
plt.figtext(0.3, 0.9,"Original Bboxes", va="top", ha="center", size=25)
plt.figtext(0.73, 0.9,"Non-max Suppression", va="top", ha="center", size=25)
plt.savefig('preprocess_output.png', bbox_inches='tight')
plt.show()

#====Building DATASET========
random.seed(42)
random.shuffle(imagepaths)
train_len = round(0.75*len(imagepaths))
train_paths = imagepaths[:train_len]
val_paths = imagepaths[train_len:]
print("Split Counts\nTrain Images: {0}\nVal Images: {1}" .format(len(train_paths), len(val_paths)))

now = datetime.datetime.now()
annodata = dict(
    info=dict(
        description=None,
        url=None,
        version=None,
        year=now.year,
        contributor=None,
        date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
    ),
    licenses=[dict(
        url=None,
        id=0,
        name=None,
    )],
    images=[
        # license, url, file_name, height, width, date_captured, id
    ],
    type='instances',
    annotations=[
        # segmentation, area, iscrowd, image_id, bbox, category_id, id
    ],
    categories=[
        # supercategory, id, name
    ],
)

class_name_to_id = {}
for i, each_label in enumerate(labels):
    class_id = i
    class_name = each_label
    class_name_to_id[class_name] = class_id
    annodata['categories'].append(dict(
        supercategory=None,
        id=class_id,
        name=class_name,
    ))

train_output_dir = "./train_images"
val_output_dir = "./val_images"

if not osp.exists(train_output_dir):
    os.makedirs(train_output_dir)
    print('Coco Train Image Directory:', train_output_dir)
    
if not osp.exists(val_output_dir):
    os.makedirs(val_output_dir)
    print('Coco Val Image Directory:', val_output_dir)
    
train_out_file = './train_annotations.json'

data_train = annodata.copy()
data_train['images'] = []
data_train['annotations'] = []

iou_thr = 0.5
skip_box_thr = 0.0001
viz_images = []

for i, path in tqdm(enumerate(train_paths)):
    img_array  = cv2.imread(path)
    image_basename = Path(path).stem
#     print(f"(\'{image_basename}\', \'{path}\')")
    
    ## Copy Image 
    shutil.copy2(path, train_output_dir)
    
    ## Add Images to annotation
    data_train['images'].append(dict(
        license=0,
        url=None,
        file_name=os.path.join('train_images', image_basename+'.jpg'),
        height=img_array.shape[0],
        width=img_array.shape[1],
        date_captured=None,
        id=i
    ))
    
    img_annotations = data[data.image_id==image_basename]
    boxes_viz = img_annotations[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().tolist()
    labels_viz = img_annotations['class_id'].to_numpy().tolist()
    
    ## Visualize Original Bboxes every 500th
    if (i%500==0):
        img_before = img_array.copy()
        for box, label in zip(boxes_viz, labels_viz):
            x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])
            color = label2color[int(label)]
            img_before = draw_bbox(img_before, list(np.int_(box)), viz_labels[label], color)
        viz_images.append(img_before)
    
    boxes_list = []
    scores_list = []
    labels_list = []
    weights = []
    
    boxes_single = []
    labels_single = []

    cls_ids = img_annotations['class_id'].unique().tolist()
    
    count_dict = Counter(img_annotations['class_id'].tolist())

    for cid in cls_ids:
        ## Performing Fusing operation only for multiple bboxes with the same label
        if count_dict[cid]==1:
            labels_single.append(cid)
            boxes_single.append(img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().squeeze().tolist())

        else:
            cls_list =img_annotations[img_annotations.class_id==cid]['class_id'].tolist()
            labels_list.append(cls_list)
            bbox = img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
            
            ## Normalizing Bbox by Image Width and Height
            bbox = bbox/(img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0])
            bbox = np.clip(bbox, 0, 1)
            boxes_list.append(bbox.tolist())
            scores_list.append(np.ones(len(cls_list)).tolist())
            weights.append(1)
    
    ## Perform WBF
    boxes, scores, box_labels = weighted_boxes_fusion(boxes_list=boxes_list, scores_list=scores_list,
                                                  labels_list=labels_list, weights=weights,
                                                  iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    
    boxes = boxes*(img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0])
    boxes = boxes.round(1).tolist()
    box_labels = box_labels.astype(int).tolist()
    boxes.extend(boxes_single)
    box_labels.extend(labels_single)
    
    img_after = img_array.copy()
    for box, label in zip(boxes, box_labels):
        x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])
        area = round((x_max-x_min)*(y_max-y_min),1)
        bbox =[
                round(x_min, 1),
                round(y_min, 1),
                round((x_max-x_min), 1),
                round((y_max-y_min), 1)
                ]
        
        data_train['annotations'].append(dict( id=len(data_train['annotations']), image_id=i,
                                            category_id=int(label), area=area, bbox=bbox,
                                            iscrowd=0))
        
    ## Visualize Bboxes after operation every 500th
    if (i%500==0):
        img_after = img_array.copy()
        for box, label in zip(boxes, box_labels):
            color = label2color[int(label)]
            img_after = draw_bbox(img_after, list(np.int_(box)), viz_labels[label], color)
        viz_images.append(img_after)

plot_imgs(viz_images, cmap=None)
plt.figtext(0.3, 0.9,"Original Bboxes", va="top", ha="center", size=25)
plt.figtext(0.73, 0.9,"WBF", va="top", ha="center", size=25)
plt.show()
               
with open(train_out_file, 'w') as f:
    json.dump(data_train, f, indent=4)
    
## Setting the output annotations json file path
val_out_file = './val_annotations.json'

data_val = annodata.copy()
data_val['images'] = []
data_val['annotations'] = []

viz_images = []

for i, path in tqdm(enumerate(val_paths)):
    img_array  = cv2.imread(path)
    image_basename = Path(path).stem
#     print(f"(\'{image_basename}\', \'{path}\')")
    
    ## Copy Image 
    shutil.copy2(path, val_output_dir)
    
    ## Add Images to annotation
    data_val['images'].append(dict(
        license=0,
        url=None,
        file_name=os.path.join('val_images', image_basename+'.jpg'),
        height=img_array.shape[0],
        width=img_array.shape[1],
        date_captured=None,
        id=i
    ))
    
    img_annotations = data[data.image_id==image_basename]
    boxes_viz = img_annotations[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().tolist()
    labels_viz = img_annotations['class_id'].to_numpy().tolist()
    
    ## Visualize Original Bboxes every 500th
    if (i%500==0):
        img_before = img_array.copy()
        for box, label in zip(boxes_viz, labels_viz):
            x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])
            color = label2color[int(label)]
            img_before = draw_bbox(img_before, list(np.int_(box)), viz_labels[label], color)
        viz_images.append(img_before)
    
    boxes_list = []
    scores_list = []
    labels_list = []
    weights = []
    
    boxes_single = []
    labels_single = []

    cls_ids = img_annotations['class_id'].unique().tolist()
    
    count_dict = Counter(img_annotations['class_id'].tolist())
    for cid in cls_ids:
        ## Performing Fusing operation only for multiple bboxes with the same label
        if count_dict[cid]==1:
            labels_single.append(cid)
            boxes_single.append(img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().squeeze().tolist())

        else:
            cls_list =img_annotations[img_annotations.class_id==cid]['class_id'].tolist()
            labels_list.append(cls_list)
            bbox = img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
            
            ## Normalizing Bbox by Image Width and Height
            bbox = bbox/(img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0])
            bbox = np.clip(bbox, 0, 1)
            boxes_list.append(bbox.tolist())
            scores_list.append(np.ones(len(cls_list)).tolist())
            weights.append(1)
            
    ## Perform WBF
    boxes, scores, box_labels = weighted_boxes_fusion(boxes_list=boxes_list, scores_list=scores_list,
                                                  labels_list=labels_list, weights=weights,
                                                  iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    
    boxes = boxes*(img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0])
    boxes = boxes.round(1).tolist()
    box_labels = box_labels.astype(int).tolist()
    boxes.extend(boxes_single)
    box_labels.extend(labels_single)
    
    img_after = img_array.copy()
    for box, label in zip(boxes, box_labels):
        x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])
        area = round((x_max-x_min)*(y_max-y_min),1)
        bbox =[
                round(x_min, 1),
                round(y_min, 1),
                round((x_max-x_min), 1),
                round((y_max-y_min), 1)
                ]
        
        data_val['annotations'].append(dict( id=len(data_val['annotations']), image_id=i,
                                            category_id=int(label), area=area, bbox=bbox,
                                            iscrowd=0))
        
    ## Visualize Bboxes after operation
    if (i%500==0):
        img_after = img_array.copy()
        for box, label in zip(boxes, box_labels):
            color = label2color[int(label)]
            img_after = draw_bbox(img_after, list(np.int_(box)), viz_labels[label], color)
        viz_images.append(img_after)
        
plot_imgs(viz_images, cmap=None)
plt.figtext(0.3, 0.9,"Original Bboxes", va="top", ha="center", size=25)
plt.figtext(0.73, 0.9,"WBF", va="top", ha="center", size=25)
plt.show()
               
with open(val_out_file, 'w') as f:
    json.dump(data_val, f, indent=4)
    
print("Number of Images in the Train Annotations File:", len(data_train['images']))
print("Number of Bboxes in the Train Annotations File:", len(data_train['annotations']))

print("Number of Images in the Val Annotations File:", len(data_val['images']))
print("Number of Bboxes in the Val Annotations File:", len(data_val['annotations']))