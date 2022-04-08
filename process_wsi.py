from pathlib import Path
from typing import Dict

import torch
import torchvision
import openslide
from util.nms_WSI import nms
from detection import MyMitosisDetection
import numpy as np
import skimage.io
import random
from time import perf_counter
import matplotlib.pyplot as plt

from skimage.measure import label, regionprops
from skimage.color import label2rgb, rgb2hsv
from skimage import morphology
from skimage import filters
import csv
import os.path
import glob
import time
# import logging
from tqdm import tqdm
import pickle

def load_model():
    
    path_model = "./model_weights/RetinaNetDA.pth"
    size = 256
    batchsize = 20
    detect_thresh = 0.64
    nms_thresh = 0.4
    level = 0
            

    md = MyMitosisDetection(path_model, size, batchsize, detect_threshold=detect_thresh, nms_threshold=nms_thresh)
    load_success = md.load_model()
    if load_success:
        print("Successfully loaded model.")
    return md


def find_main_tissue(slide):
    # extract overview image from wholeslide
    levels = slide.level_dimensions

    k = len(levels)
    l = k - 3 # arbitrary level. Might fail when working with slides that have too few levels.
    im_size = levels[l]

    start = (0, 0)
    rgba_im = slide.read_region(start, l, im_size)

    # convert from RGBA to RGB
    tissue_im = np.array(rgba_im.convert('RGB'))

    # convert rgb2hsv
    HSV = rgb2hsv(tissue_im)

    # median filtering of H and S channel
    disk = morphology.disk(13.) # arbitrary element size. It works empirically well
    H_im = filters.median(HSV[:, :, 0], disk)
    S_im = filters.median(HSV[:, :, 1], disk)

    # Select only the areas that are scanned
    maskH = H_im > 0.
    maskS = S_im > 0.

    # Otsu thresholding
    valH = filters.threshold_otsu(H_im[maskH])
    valS = filters.threshold_otsu(S_im[maskS])

    Hbin = H_im > valH
    Sbin = S_im > valS

    # combine H and S channel to get final mask
    final_mask = Hbin + Sbin

    return final_mask, l


def analyze_wholeslides(weights, slides, patch_size, export_location):
    """

    :param weights:
    :param slide:
    :param patch_size:
    :param export_location:
    :return:
    """

    md_model = load_model()    
    nms_thresh = 0.4

    for slide in slides:
        print(f'Started analyzing slide {slide}')
        start_slide = time.time()

        wholeslide = openslide.OpenSlide(slide)
        mask, level = find_main_tissue(slide)
        downsample = wholeslide.level_downsamples[level]
        x_size = int(((slide.dimensions[0] - patch_size) / stride) + 1)
        y_size = int(((slide.dimensions[1] - patch_size) / stride) + 1)
        
        
        processing_coordinates = split_tissue(wholeslide, patch_size)
        print("Found {} patches".format(len(processing_coordinates)))
        #pool = Pool(MAX_WORKERS)
        results = []
        for patch in tqdm(processing_coordinates):

            im = wholeslide.read_region((int(patch[1]), int(patch[0])), 0, (patch[3], patch[2]))
            tile = im.convert('RGB')
            hpf = np.asarray(tile)
            # logging.debug(hpf.shape)

            # tile_time = time.time()
            #        
            with torch.inference_mode():
                result_boxes = md_model.process_image(hpf)
#             # perform nms per image:
#         # print("All computations done, nms as a last step")
                result_boxes = nms(result_boxes, nms_thresh)
                candidates = list()
                for i, detection in enumerate(result_boxes):
                    # our prediction returns x_1, y_1, x_2, y_2, prediction, score -> transform to center coordinates
                    x_1, y_1, x_2, y_2, prediction, score = detection
                    coord = tuple(((x_1 + x_2) / 2, (y_1 + y_2) / 2))
                    candidates.append(coord)
                result = [{"detection": [x, y]} for x, y in candidates]
#  
            

            # logging.debug("Delta time for tile {} is {}".format(patch, delta_tile_time))
            if len(result) > 0:
                results.append([patch, result])
                print("Detected points on slide {} \n Points: {}".format(os.path.basename(slide), result))
                with open('{}/{}_detected_points.csv'.format(export_location, os.path.basename(slide)), 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([patch, result])
        with open('{}/{}_pickled_results.p'.format(export_location, os.path.basename(slide)), 'wb') as p_file:
            pickle.dump(results, p_file)
        print('Time to analyze {} - {}'.format(slide, time.time() - start_slide))
        wholeslide.close()


def main():
    pass
    

if __name__ == "__main__":


    #slides = glob.glob("/home/nikolas/projects/MitosisDetector/data/test_examples/*.ndpi")
    export_location = '/hpc/dla_patho/data/mitosis_validation/mitosis_results_retinaNet/'
    slides_location = '/hpc/dla_patho/data/mitosis_validation/WSI/'

    slides = glob.glob(os.path.join(slides_location, '*.ndpi'))
    #slides = glob.glob("/mnt/S/Research/mitosis/validation_set/*.ndpi")
    # slides = []
    # slides.append('/mnt/S/Research/mitosis/PACS dataset/excision/T19-02537_II7_HE  1_989498/LMS-8-6743335 - 2019-02-25 19.27.15.ndpi')


    #slides = glob.glob("/mnt/S/Research/mitosis/validation_set/*.ndpi")
    # logPath, fileName = os.path.split(logfile)
    # logging.root.handlers = []
    # logging.basicConfig(
    #     format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    #     level=logging.DEBUG,
    #     handlers=[
    #         # logging.FileHandler("{0}/{1}".format(logPath, fileName)),
    #         logging.FileHandler("{0}".format("analysis.log")),
    #         logging.StreamHandler()
    #     ])

    # Locating model weights
    # path_weights_npz = "./weights/weights_model_CA.npz"  # -- Operating Point: 0.770
    # operatingPoint_threshold = 0.770
    

   

    #path_weights_npz = "./weights/weights_model_CA_DANN.npz"  # -- Operating Point: 0.816
    #operatingPoint_threshold = 0.816
    
    # path_weights_npz = "weights/weights_model_8rotGCNN_15AL.npz" #-- Operating Point: 0.647
    # operatingPoint_threshold = 0.793

    start_time = time.time()
    print("\n Begin analysis")
    # print("Weights: {}".format(path_weights_npz))    
    
    # logging.info("\n Begin analysis")
    # logging.info("Weights: {}".format(path_weights_npz))    
    # operatingPoint_threshold = 0.7

    #wholeslide = openslide.OpenSlide(slide)

    #processing_patches = split_tissue(wholeslide, 1024)
    #print(processing_patches)

    # processes_for_slides = []
    # for slide in slides:
    #     p = multiprocessing.Process(target=analyze_wholeslide, args=(path_weights_npz, slide, 1024, export_location))
    #     p.start()
    #     processes_for_slides.append(p)
    #
    # for p in processes_for_slides:
    #     p.join()
    
    # for slide in slides:
    #     slide_time = time.time() - start_time     
    
    analyze_wholeslides(slides, 1024, export_location)
    elapsed_time = time.time() - start_time
    print("Total time for analysis:  {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))



            
    # candidates = list()
    # for i, detection in enumerate(result_boxes):
    #     # our prediction returns x_1, y_1, x_2, y_2, prediction, score -> transform to center coordinates
    #     x_1, y_1, x_2, y_2, prediction, score = detection
    #     coord = tuple(((x_1 + x_2) / 2, (y_1 + y_2) / 2))
    #     print(coord)
    #     candidates.append(coord)

            # For the test set, we expect the coordinates in millimeters - this transformation ensures that the pixel
            # coordinates are transformed to mm - if resolution information is available in the .tiff image. If not,
            # pixel coordinates are returned.
            # world_coords = input_image.TransformContinuousIndexToPhysicalPoint(
            #     [c for c in reversed(coord)]
            # )
            # candidates.append(tuple(reversed(world_coords)))
            

        # Note: We expect you to perform thresholding for your predictions. For evaluation, no additional thresholding
        # will be performed
        # result = [{"point": [x, y, 0]} for x, y in candidates]


    # def process_batch(self):
    #     pass
    #     # load image

    #     # 