import argparse
import csv
import os
import os.path as osp
import time
from glob import glob

# from mean_average_precision.detection_map import DetectionMAP
import cv2
import numpy as np
import torch
from tqdm import tqdm

from head_detection.data import (cfg_mnet, cfg_res50, cfg_res50_4fpn,
                                 cfg_res152, ch_anchors, combined_anchors,
                                 headhunt_anchors, sh_anchors)
from head_detection.models.head_detect import customRCNN
from head_detection.utils import my_load, get_state_dict, plot_ims, to_torch
from head_detection.vision.utils import init_distributed_mode

# try:
#     from scipy.misc import imread, imsave
# except ImportError:
#     from scipy.misc.pilutil import imread

from PIL import Image
from albumentations.pytorch import ToTensor

parser = argparse.ArgumentParser(description='Testing script')
parser.add_argument('--test_dataset', help='Dataset .txt file')
parser.add_argument('--pretrained_model', help='resume net for retraining')
parser.add_argument('--plot_folder', help='Location to plot results on images')
parser.add_argument('--image_folder', help='Location to the image folder')

parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')

parser.add_argument('--benchmark', default='Combined', help='Benchmark for training/validation')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--min_size', default=720, type=int, help='Optionally plot first N images of test')
parser.add_argument('--max_size', default=1280, type=int, help='Optionally plot first N images of test')

parser.add_argument('--ext', default='.jpg', type=str, help='Image file extensions')
parser.add_argument('--outfile', help='Location to save results in mot format')

parser.add_argument('--backbone', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--context', help='Whether to use context model')
parser.add_argument('--use_deform', default=False, type=bool, help='Use Deformable SSH')
parser.add_argument('--det_thresh', default=0.3, type=float, help='Number of workers used in dataloading')
parser.add_argument('--default_filter', default=False, type=bool, help='Use old filters')
parser.add_argument('--soft_nms', default=False, type=bool, help='Use soft nms?')
parser.add_argument('--upscale_rpn', default=False, type=bool, help='Upscale RPN feature maps')

args = parser.parse_args()

##################################
## Set device and config ##########
##################################
if torch.cuda.is_available():
    device = torch.device('cuda')
cfg = None
if args.backbone == "mobile0.25":
    cfg = cfg_mnet
elif args.backbone == "resnet50":
    cfg = cfg_res50_4fpn
elif args.backbone == "resnet152":
    cfg = cfg_res152
else:
    raise ValueError("Invalid configuration")


def create_model(combined_cfg):
    kwargs = {}
    kwargs['min_size'] = args.min_size
    kwargs['max_size'] = args.max_size
    kwargs['box_score_thresh'] = args.det_thresh
    kwargs['box_nms_thresh'] = 0.5
    kwargs['box_detections_per_img'] = 300 # increase max det to max val in our benchmark
    model = customRCNN(cfg=combined_cfg, use_deform=args.use_deform,
                       context=args.context, default_filter=args.default_filter,
                       soft_nms=args.soft_nms, upscale_rpn=args.upscale_rpn,
                       **kwargs).cuda()
    return model


@torch.no_grad()
def test():
    # print("Testing FPN. On single GPU without Parallelism")
    cpu_device = torch.device("cpu")

    # Set benchmark related parameters
    if args.benchmark == 'ScutHead':
        combined_cfg = {**cfg, **sh_anchors}
    elif args.benchmark == 'CHuman':
        combined_cfg = {**cfg, **ch_anchors}
    elif args.benchmark == 'Combined':
        combined_cfg = {**cfg, **combined_anchors}
    else:
        raise ValueError("New dataset has to be registered")
    model = create_model(combined_cfg)

    # state_dict = torch.load(args.pretrained_model)
    # for k, v in state_dict.items():
    #     print(k) 

    # for k, v in model.state_dict().items():
        # print(k) 

    # new_state_dict = get_state_dict(model, args.pretrained_model)
    # model.load_state_dict(new_state_dict, strict=True)
    checkpoint = torch.load(args.pretrained_model)
    model = my_load(model, checkpoint['model_state_dict'], 
        only_backbone=False)

    model = model.eval()

    video_frame_path = args.image_folder
    frame_name_list = os.listdir(video_frame_path)
    frame_name_list.sort()
    detection_list = []
    for name in tqdm(frame_name_list, ascii=True, position=0):
        img_path = os.path.join(video_frame_path, name)
        img = Image.open(img_path)
        np_img = np.array(img)
        transf = ToTensor()
        img_tensor = transf(image=np_img)['image'].to(torch.device("cuda"))
        img_tensor = torch.unsqueeze(img_tensor, 0)
        outputs = model(img_tensor)
        # print([outputs[0].keys()])
        # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        out_dict = {'boxes': outputs[0]['boxes'].cpu(), 'scores': outputs[0]['scores'].cpu()}
        # plot_img = plot_ims(np_img.transpose(2,0,1), out_dict['boxes'].cpu().numpy()) 

        filename, _ = os.path.splitext(name)
        frame_idx = filename.split("_")[-1]
        pred_boxes = out_dict['boxes'].cpu().numpy()
        scores = out_dict['scores'].cpu().numpy()
        for b_id, (box, score) in enumerate(zip(pred_boxes, scores)):
            (startX, startY, endX, endY) = box
            detection_list.append((int(frame_idx), startX, startY, endX - startX, endY - startY, score))
            # print(b_id, startX, startY, endX, endY, score)
        
        # out_path = os.path.join("/content", os.path.basename(img_path))
        # cv2.imwrite(out_path, plot_img)

    out_path = os.path.join(os.path.dirname(video_frame_path), "det")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    out_path = os.path.join(os.path.dirname(video_frame_path), "det", "det.txt")
    with open(out_path, mode="w") as f:
        out_list = []
        for i, det in enumerate(detection_list):
            out_list.append(f"{det[0]},-1,{det[1]},{det[2]},{det[3]},{det[4]},{det[5] * 100},1,1,1")
                
        for line in out_list:
            f.write("%s\n" % line)


if __name__ == '__main__':
    test()