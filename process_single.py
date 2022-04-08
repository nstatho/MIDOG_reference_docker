import logging
import torch
from queue import Queue, Empty
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
from util.nms_WSI import nms_patch, nms
from util.object_detection_helper import create_anchors, process_output, rescale_box
from fastai.vision.learner import create_body
from fastai.vision import models

from model import RetinaNetDA


class MyMitosisDetection:
    def __init__(self, path_model, size, batchsize, detect_threshold = 0.64, nms_threshold = 0.4):

        # network parameters
        self.detect_thresh = detect_threshold
        self.nms_thresh = nms_threshold
        encoder = create_body(models.resnet18, False, -2)
        scales = [0.2, 0.4, 0.6, 0.8, 1.0]
        ratios = [1]
        sizes = [(64, 64), (32, 32), (16, 16)]
        self.model = RetinaNetDA.RetinaNetDA(encoder, n_classes=2, n_domains=4,  n_anchors=len(scales) * len(ratios),sizes=[size[0] for size in sizes], chs=128, final_bias=-4., n_conv=3)
        self.path_model = path_model
        self.size = size
        self.batchsize = batchsize
        self.mean = None
        self.std = None
        self.anchors = create_anchors(sizes=sizes, ratios=ratios, scales=scales)
        self.device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

    def load_model(self):
        self.mean = torch.FloatTensor([0.7481, 0.5692, 0.7225]).to(self.device)  # state['data']['normalize']['mean']
        self.std = torch.FloatTensor([0.1759, 0.2284, 0.1792]).to(self.device)  # state['data']['normalize']['std']

        if torch.cuda.is_available():
            print("Model loaded on CUDA")
            self.model.load_state_dict(torch.load(self.path_model))
        else:
            print("Model loaded on CPU")
            self.model.load_state_dict(torch.load(self.path_model, map_location='cpu'))

        self.model.to(self.device)

        logging.info("Model loaded. Mean: {} ; Std: {}".format(self.mean, self.std))
        return True

    def process_image(self, input_image):
        self.model.eval()
        image_boxes = []                
        cur_patch = input_image.transpose(2,0,1)[0:3] / 255.
        cur_patch = np.expand_dims(cur_patch, axis=0)
        torch_batch = torch.from_numpy(cur_patch.astype(np.float32, copy=False)).to(self.device)
        
        class_pred_batch, bbox_pred_batch, domain,_ = self.model(torch_batch)

        for b in range(torch_batch.shape[0]):
            cur_class_pred = class_pred_batch[b]
            cur_bbox_pred = bbox_pred_batch[b]
            cur_patch_boxes = self.postprocess_patch(cur_bbox_pred, cur_class_pred, 0, 0)
            if len(cur_patch_boxes) > 0:
                image_boxes += cur_patch_boxes

        return np.array(image_boxes)
    
    

    def postprocess_patch(self, cur_bbox_pred, cur_class_pred, x_real, y_real):
        cur_patch_boxes = []

        for clas_pred, bbox_pred in zip(cur_class_pred[None, :, :], cur_bbox_pred[None, :, :], ):
            modelOutput = process_output(clas_pred, bbox_pred, self.anchors, self.detect_thresh)
            bbox_pred, scores, preds = [modelOutput[x] for x in ['bbox_pred', 'scores', 'preds']]

            if bbox_pred is not None:
                # Perform nms per patch to reduce computation effort for the whole image (optional)
                to_keep = nms_patch(bbox_pred, scores, self.nms_thresh)
                bbox_pred, preds, scores = bbox_pred[to_keep].cpu(), preds[to_keep].cpu(), scores[
                    to_keep].cpu()

                t_sz = torch.Tensor([[self.size, self.size]]).float()

                bbox_pred = rescale_box(bbox_pred, t_sz)

                for box, pred, score in zip(bbox_pred, preds, scores):
                    y_box, x_box = box[:2]
                    h, w = box[2:4]

                    cur_patch_boxes.append(
                        np.array([x_box + x_real, y_box + y_real,
                                  x_box + x_real + w, y_box + y_real + h,
                                  pred, score]))

        return cur_patch_boxes