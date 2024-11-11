import torch.optim as optim
from Vgg.Vgg19 import Vgg19
import torch.nn as nn
import torch
from utils import *
from loss import *
import numpy


def run_attack(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_image, style_image, content_seg, style_seg = load_imgs_from_path(args)
    content_image = content_image.to(device)
    style_image = style_image.to(device)
    content_seg = content_seg.to(device)
    style_seg = style_seg.to(device)

    content_masks, style_masks = load_seg(content_seg, style_seg)
    input_image = content_image.clone().requires_grad_(True)

    vgg_const = Vgg19(requires_grad=False).to(device)
    vgg_const.eval()

    resized_content_image = nn.functional.interpolate(content_image, size=(224, 224), mode='bilinear', align_corners=False)
    prob = vgg_const(resized_content_image).softmax
    pred = print_prob(prob.cpu().detach().numpy()[0], './synset.txt')
    args.true_label = torch.argmax(prob).item()
    ori = get_class('./synset.txt',args.true_label)
    vgg_const.eval()
    content_fv = vgg_const(content_image)
    content_layer_const = content_fv.conv4_2
    style_layers_const_c = [content_fv[i] for i in [0,1,2,3,5]]

    vgg_const.eval()
    style_fvs = vgg_const(style_image)
    style_layers_const = [style_fvs[i] for i in [0,1,2,3,5]]

    vgg_var = Vgg19(requires_grad=False).to(device)
    optimizer = optim.Adam([input_image], lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    
    for i in range(0, args.max_iter+1):
        optimizer.zero_grad()
        vgg_var.eval()
        var_fv = vgg_var(input_image)
        style_layers_var = [var_fv[i] for i in [0,1,2,3,5]]
        content_layer_var = vgg_var(input_image).conv4_2
        
        content_masks_copy = [mask.clone() for mask in content_masks]
        style_masks_copy = [mask.clone() for mask in style_masks]
        loss_content = content_loss(content_layer_const, content_layer_var, float(args.content_weight))
        loss_styles_list = style_loss(vgg_var.cnn_struct, style_layers_const, style_layers_const_c, style_layers_var, content_masks_copy, style_masks_copy, float(args.style_weight))
        loss_style = 0.0
        for loss in loss_styles_list:
            loss_style += loss

        pred = vgg_var.logits
        if args.targeted_attack:
            loss_attack = targeted_attack_loss(pred=pred, orig_pred=args.true_label, target=args.target_label,
                                        weight=args.attack_weight)
        else:
            loss_attack = untargeted_attack_loss(pred=pred, orig_pred=args.true_label, weight=args.attack_weight)

        #loss_tv = total_variation_loss(input_image, float(args.tv_weight))
        total_loss = loss_content + loss_style + loss_attack #loss_tv + 

        total_loss.backward()
        optimizer.step()

        if i % args.save_iter == 0:
            suc = 'non'
            if args.targeted_attack == 1:
                if torch.argmax(pred).item() == args.target_label:
                    suc = 'suc'
            else:
                if torch.argmax(pred).item() != args.true_label:
                    suc = 'suc'
            print(f"step:{i}")
            # print('\tTV loss: {}'.format(loss_tv))
            print('\tContent loss: {}'.format(loss_content))
            print('\tStyle loss: {}'.format(loss_style))
            print('\tAttack loss: {}'.format(loss_attack))
            print('\tTotal loss: {}'.format(total_loss))
            pred_label_inx = torch.argmax(var_fv.softmax,dim=1).item()
            pred_probability = var_fv.softmax[0, pred_label_inx].item()
            obj = get_class('./synset.txt',pred_label_inx)
            save_result(input_image, os.path.join(args.result_dir, f'{suc}_{i}_{ori}_{obj}_{pred_probability}.jpg'))
            
    pred_label_inx = torch.argmax(var_fv.softmax,dim=1).item()
    pred_probability = var_fv.softmax[0, pred_label_inx].item()
    obj = get_class('./synset.txt',pred_label_inx)
    ori = get_class('./synset.txt',args.true_label)
    save_result(input_image, f"{args.result_dir}/final_{ori}_{obj}_{pred_probability}.jpg")