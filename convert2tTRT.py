import torch
from torch2trt import torch2trt
import torchvision.transforms as transforms
from util.nms_WSI import nms_patch, nms
from util.object_detection_helper import create_anchors, process_output, rescale_box
from fastai.vision.learner import create_body
from fastai.vision import models

from model import RetinaNetDA

path_model = "./model_weights/RetinaNetDA.pth"
size = 512
batchsize = 10
detect_thresh = 0.64
nms_thresh = 0.4

encoder = create_body(models.resnet18, False, -2)
scales = [0.2, 0.4, 0.6, 0.8, 1.0]
ratios = [1]
sizes = [(64, 64), (32, 32), (16, 16)]
model = RetinaNetDA.RetinaNetDA(encoder, n_classes=2, n_domains=4,  n_anchors=len(scales) * len(ratios),sizes=[size[0] for size in sizes], chs=128, final_bias=-4., n_conv=3)

anchors = create_anchors(sizes=sizes, ratios=ratios, scales=scales)
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

mean = torch.FloatTensor([0.7481, 0.5692, 0.7225]).to(device)  # state['data']['normalize']['mean']
std = torch.FloatTensor([0.1759, 0.2284, 0.1792]).to(device)  # state['data']['normalize']['std']

if torch.cuda.is_available():
    print("Model loaded on CUDA")
    model.load_state_dict(torch.load(path_model))
else:
    print("Model loaded on CPU")
    model.load_state_dict(torch.load(path_model, map_location='cpu'))

model.to(device)

