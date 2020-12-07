from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import torch
import numpy as np
import cv2
import random
import sys
import time
import os

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()


# def get_styles(masks):
#     styles = []
#     newsize = (masks.shape[2], masks.shape[1])
#
#     img1 = Image.open('styles/style0.jpg').resize(newsize)
#     img2 = Image.open('styles/style1.jpg').resize(newsize)
#
#     img1 = np.asarray(img1)
#     img2 = np.asarray(img2)
#
#     styles.append(img1)
#     styles.append(img2)
#
#     styles = np.array(styles)
#     return styles


def combine_masks(styles, style_id, masks):
    masks = np.stack((masks,) * 3, axis=-1)
    combine_list = []
    s = [styles[i] for i in style_id]
    for i in range(len(s)):
        mask = masks[i]
        style = s[i]
        template = np.multiply(mask, style)
        template = template.astype(np.uint8)
        combine_list.append(template)
    return combine_list


def random_colour_masks(image):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = [0, 255, 0]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def get_prediction(img_path, threshold):
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img).to(device)
    pred = model([img])

    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]

    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    masks = masks[:pred_t + 1]
    frame = np.zeros((masks.shape[1], masks.shape[2]))
    for mask in masks:
        frame[mask == 1] = 1
    bg = 1 - frame
    bg = np.array([bg])
    masks = np.append(masks, bg, axis=0)
    return masks


def select_styles(img_path, num_styles, threshold):
    masks = get_prediction(img_path, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    style_ids = []
    for i in range(len(masks)):
        rgb_mask = random_colour_masks(masks[i])
        temp_img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        plt.figure(figsize=(10, 18))
        plt.imshow(temp_img)
        plt.show()
        # in case img show delayed
        time.sleep(2)
        # prompt style selection
        while True:
            if i < len(masks) - 1:
                num = input(
                    "Please enter a style id (0 to {0}) for person/obj {1} (-1 to quit):".format(num_styles - 1, i))
            else:
                num = input(
                    "Please enter a style id (0 to {0}) for the background (-1 to quit):".format(num_styles - 1))
            try:
                id = int(num)
                if num_styles - 1 >= id >= 0:
                    print("Style {0} selected".format(id, i))
                    style_ids.append(id)
                    break
                elif id == -1:
                    print("Program terminated.")
                    sys.exit(0)
                else:
                    print("Error. Style id is not valid.")
            except ValueError:
                print("Error. Style id should be an integer.")
    print("Style selection completed.")
    return style_ids

def instance_segmentation_api_AMG(img_path, num_styles, threshold=0.6):
    masks = get_prediction(img_path, threshold)
    #styles = get_styles(masks)
    selected = select_styles(img_path, num_styles, threshold)
    combine_list = combine_masks(styles, selected, masks)

    canvas = np.zeros_like(styles[0]).astype(np.uint8)
    for obj in combine_list:
        canvas = cv2.add(canvas, obj)

    plt.figure(figsize=(10, 18))
    plt.imshow(canvas)
    plt.show()


instance_segmentation_api_AMG('hoofer.jpg', 2, 0.9)



