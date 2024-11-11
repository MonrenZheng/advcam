import argparse
import os
from attack import run_attack

parser = argparse.ArgumentParser()
parser.add_argument("--content_image_path", type=str, default='./data/content/stop-sign/stop-sign.jpg')
parser.add_argument("--style_image_path", type=str, default='./data/style/stop-sign/dir1.jpg')
parser.add_argument("--content_seg_path",   dest='content_seg_path',    nargs='?',
                    help="Path to the style segmentation",default='./data/content-mask/stop-sign.jpg')
parser.add_argument("--style_seg_path",     dest='style_seg_path',      nargs='?',
                    help="Path to the style segmentation",default='./data/style-mask/dir1.jpg')
parser.add_argument("--result_dir", type=str, default='./Result/sample6')
parser.add_argument("--target_label", type=int, default=77)  # 默认目标类别
parser.add_argument("--targeted_attack", action="store_true",default=True)  # 针对性的攻击
parser.add_argument("--content_weight", type=float, default=5e0)
parser.add_argument("--style_weight", type=float, default=3e3)
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