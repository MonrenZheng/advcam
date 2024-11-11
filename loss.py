import torch
from utils import *
import math
import torch.nn as nn


def content_loss(const_layer, var_layer, weight):
    return torch.mean((const_layer - var_layer) ** 2) * weight

def style_loss(cnn_structure, const_layers, content_const_layers, var_layers, content_segs, style_segs, weight):
    loss_styles = []
    layer_index = 0
    content_seg_height, content_seg_width = content_segs[0].size(2), content_segs[0].size(3)
    style_seg_height, style_seg_width = style_segs[0].size(2), style_segs[0].size(3)

    for layer_name in cnn_structure:
        with torch.no_grad():
            # downsampling segmentation
            if "pool" in layer_name:
                content_seg_width, content_seg_height = int(math.ceil(content_seg_width / 2)), int(math.ceil(content_seg_height / 2))
                style_seg_width, style_seg_height = int(math.ceil(style_seg_width / 2)), int(math.ceil(style_seg_height / 2))

                for i in range(len(content_segs)):
                    content_segs[i] = nn.functional.interpolate(content_segs[i], size=(content_seg_height, content_seg_width), mode='bilinear', align_corners=False)
                    style_segs[i] = nn.functional.interpolate(style_segs[i], size=(style_seg_height, style_seg_width), mode='bilinear', align_corners=False)

            elif "conv" in layer_name:
                for i in range(len(content_segs)):
                    content_segs[i] = nn.functional.avg_pool2d(nn.functional.pad(content_segs[i], (1, 1, 1, 1), "constant", 0), kernel_size=3, stride=1, padding=0)
                    style_segs[i] = nn.functional.avg_pool2d(nn.functional.pad(style_segs[i], (1, 1, 1, 1), "constant", 0), kernel_size=3, stride=1, padding=0)
        
        if layer_name in ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1']:
            # print("successful!!!")
            # print("Setting up style layer: <{}>".format(layer_name))
            const_layer = const_layers[layer_index]  # 风格图像
            content_const_layer = content_const_layers[layer_index]  # 内容图像
            var_layer = var_layers[layer_index]  # 合成图像

            layer_index += 1

            layer_style_loss = 0.0
            for content_seg, style_seg in zip(content_segs, style_segs):
                gram_matrix_var = gram_matrix(var_layer * content_seg)
                content_mask_mean = torch.mean(content_seg)
                gram_matrix_var = torch.where(content_mask_mean > 0, gram_matrix_var / (torch.numel(var_layer) * content_mask_mean), gram_matrix_var)
                cur_style_mask_mean = torch.mean(style_seg)
                style_mask_mean = torch.where(torch.logical_and(content_mask_mean > 0, torch.eq(cur_style_mask_mean, 0.0)), torch.mean(content_seg), torch.mean(style_seg))
                cur_const_layer = torch.where(torch.logical_and(content_mask_mean > 0, torch.eq(cur_style_mask_mean, 0.0)), content_const_layer, const_layer)

                gram_matrix_const = torch.where(torch.logical_and(content_mask_mean > 0, torch.eq(cur_style_mask_mean, 0.0)), gram_matrix(content_const_layer * content_seg), gram_matrix(const_layer * style_seg))
                gram_matrix_const = torch.where(style_mask_mean > 0, gram_matrix_const / (torch.numel(cur_const_layer) * style_mask_mean), gram_matrix_const)
                diff_style_sum = torch.mean((gram_matrix_const - gram_matrix_var) ** 2) * content_mask_mean
                layer_style_loss += diff_style_sum

            loss_styles.append(layer_style_loss * weight)

    return loss_styles

def total_variation_loss(output, weight):
    tv_loss = torch.sum(
        (output[:, :, :-1, :-1] - output[:, :, :-1, 1:]) ** 2 +
        (output[:, :, :-1, :-1] - output[:, :, 1:, :-1]) ** 2) / 2.0
    return tv_loss * weight

def targeted_attack_loss(pred, orig_pred, target, weight):
    balance = 5
    orig_pred = torch.eye(1000)[orig_pred].unsqueeze(0)
    target = torch.eye(1000)[target].unsqueeze(0)
    loss1 = -torch.nn.functional.cross_entropy(pred, orig_pred)
    loss2 = torch.nn.functional.cross_entropy(pred, target)
    loss_attack = torch.sum(balance * loss2 + loss1) * weight
    return loss_attack

def untargeted_attack_loss(pred, orig_pred, weight):
    orig_pred = torch.eye(1000)[orig_pred].unsqueeze(0)
    loss1 = -torch.nn.functional.cross_entropy(pred, orig_pred)
    loss_attack = loss1
    return loss_attack * weight