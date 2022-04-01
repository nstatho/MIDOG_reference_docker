from pathlib import Path
import torch
import torchvision
import openslide
from util.nms_WSI import nms
from detection import MyMitosisDetection
import numpy as np
import skimage.io
import random
from time import perf_counter



    
path_model = "./model_weights/RetinaNetDA.pth"
size = 512
batchsize = 10
detect_thresh = 0.64
nms_thresh = 0.4
level = 0
        

md = MyMitosisDetection(path_model, size, batchsize, detect_threshold=detect_thresh, nms_threshold=nms_thresh)
load_success = md.load_model()
if load_success:
    print("Successfully loaded model.")

# path_images = Path('test')
path_images = Path('/hpc/dla_patho/home/nikolas-projects/test_data_mitosis/')


input_images = []
for image in path_images.glob('*.jpeg'):
    input_image = skimage.io.imread(image)
    input_images.append(input_image)

while True:
    t1_start = perf_counter()
    i = random.randint(0,3)
    with torch.inference_mode():
        result_boxes = md.process_image(input_images[i])
            # perform nms per image:
        # print("All computations done, nms as a last step")
        result_boxes = nms(result_boxes, nms_thresh)
        # print(result_boxes)
        candidates = list()
        for i, detection in enumerate(result_boxes):
            # our prediction returns x_1, y_1, x_2, y_2, prediction, score -> transform to center coordinates
            x_1, y_1, x_2, y_2, prediction, score = detection
            coord = tuple(((x_1 + x_2) / 2, (y_1 + y_2) / 2))
            candidates.append(coord)
        result = [{"point": [x, y, 0]} for x, y in candidates]
        print(result)
    t1_stop = perf_counter()
    print("Elapsed time:", t1_stop - t1_start)
    


    
