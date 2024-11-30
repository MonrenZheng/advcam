import argparse
import os
from attack import run_attack

parser = argparse.ArgumentParser()
parser.add_argument("--content_image_path", type=str, default='./data/content/stop-sign/stop-sign.jpg')
parser.add_argument("--style_image_path", type=str, default='./data/style/stop-sign/stop-sign.jpg')
parser.add_argument("--content_seg_path",   dest='content_seg_path',    nargs='?',
                    help="Path to the style segmentation",default='./data/content-mask/stop-sign.jpg')
parser.add_argument("--style_seg_path",     dest='style_seg_path',      nargs='?',
                    help="Path to the style segmentation",default='./data/style-mask/stop-sign.jpg')
parser.add_argument("--result_dir", type=str, default='./Result/sample1')
parser.add_argument("--target_label", type=int, default=200)  # 默认目标类别，可根据synset.txt进行自主修改
parser.add_argument("--targeted_attack", action="store_true",default=True)  # 有无目标攻击
parser.add_argument("--content_weight", type=float, default=9e0)
parser.add_argument("--style_weight", type=float, default=5e3)
parser.add_argument("--tv_weight", type=float, default=1e-3)
parser.add_argument("--attack_weight", type=float, default=5e2)
parser.add_argument("--learning_rate", type=float, default=1.0)
parser.add_argument("--max_iter", type=int, default=300)
parser.add_argument("--save_iter", type=int, default=30)

parser.add_argument("--true_label",       dest='true_label',        nargs='?', type = int,
                    help="The target label for target attack", default=8)
args = parser.parse_args()

if __name__ == "__main__":

    os.chdir(r'D:\DEMO\Deeplearning\Adverse-transfer') # 请修改到项目路径
    os.makedirs(args.result_dir, exist_ok=True)
    run_attack(args)
