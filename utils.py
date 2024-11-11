import numpy as np
import os
import torch
import torch.nn as nn
from PIL import Image

def load_imgs_from_path(args, target_size=(224, 224)):
    content_image = Image.open(args.content_image_path).convert("RGB").resize(target_size)
    content_image = np.array(content_image, dtype=np.float32)
    content_width, content_height = content_image.shape[1], content_image.shape[0]
    content_image = content_image.transpose((2, 0, 1)).reshape((1, 3, content_height, content_width))
    #print("content:",content_image)
    style_image = Image.open(args.style_image_path).convert("RGB").resize(target_size)
    style_image = np.array(style_image, dtype=np.float32)
    style_width, style_height = style_image.shape[1], style_image.shape[0]
    style_image = style_image.transpose((2, 0, 1)).reshape((1, 3, style_height, style_width))
    #print("style:",style_image)

    content_seg = Image.open(args.content_seg_path).convert("RGB").resize((content_width, content_height), resample=Image.BILINEAR)
    content_seg = np.array(content_seg, dtype=np.float32) // 245.0
    content_seg = content_seg.transpose((2, 0, 1)).reshape((1, 3, content_height, content_width))

    style_seg = Image.open(args.style_seg_path).convert("RGB").resize((style_width, style_height), resample=Image.BILINEAR)
    style_seg = np.array(style_seg, dtype=np.float32) // 245.0
    style_seg = style_seg.transpose((2, 0, 1)).reshape((1, 3, style_height, style_width))

    return torch.tensor(content_image), torch.tensor(style_image), torch.tensor(content_seg), torch.tensor(style_seg)

def load_seg(content_seg, style_seg):
    color_codes = ['UnAttack', 'Attack']

    def _extract_mask(seg, color_str):
        h, w, c = seg.shape[2], seg.shape[3], seg.shape[1]
        if color_str == "UnAttack":
            mask_r = (seg[:, 0, :, :] < 0.5).byte()
            mask_g = (seg[:, 1, :, :] < 0.5).byte()
            mask_b = (seg[:, 2, :, :] < 0.5).byte()
        elif color_str == "Attack":
            mask_r = (seg[:, 0, :, :] > 0.8).byte()
            mask_g = (seg[:, 1, :, :] > 0.8).byte()
            mask_b = (seg[:, 2, :, :] > 0.8).byte()
        return (mask_r & mask_g & mask_b).float().unsqueeze(1)

    color_content_masks = []
    color_style_masks = []
    for i in range(len(color_codes)):
        color_content_masks.append(_extract_mask(content_seg, color_codes[i]))
        color_style_masks.append(_extract_mask(style_seg, color_codes[i]))

    return color_content_masks, color_style_masks
    
def gram_matrix(activations):
    batch_size, num_channels, height, width = activations.size()
    activations = activations.view(batch_size * num_channels, height * width)
    gram_matrix = torch.mm(activations, activations.t())
    return gram_matrix

def save_result(img, str_):
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = np.uint8(np.clip(img, 0, 255.0))
    result = Image.fromarray(img)
    result.save(str_)
    
def get_class(path, inx):
    # 读取 ImageNet 类别标签文件并存储在列表中
    imagenet_classes = [x.strip() for x in open(path).readlines()]
    if inx > 1000: return None
    return imagenet_classes[inx].split(' ', 1)[1]

def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]
    pred = np.argsort(prob)[::-1]
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return pred[0], prob[pred[0]]
