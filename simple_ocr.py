#
# You can modify this files
#

import random
import argparse
import glob
import os
import cv2
import matplotlib.pyplot as plt

class HoadonOCR:

    def __init__(self):
        # Init parameters, load model here
        self.model = None
        self.labels = ['highlands', 'starbucks', 'phuclong', 'others']

    # TODO: implement find label
    def find_label(self, img):
        str = "./templates"
        img_rgb = cv2.imread(img, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        for template in self.get_templates(str):
            result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            
        plt.imshow(result, cmap='gray')
        

    def get_templates(path='./templates'):
        for f in glob.glob(os.path.join(path, '**', '*')):
            yield cv2.imread(f, cv2.IMREAD_GRAYSCALE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', type=str, help='Path to input image')
    parser.add_argument('--output_file', '-o', type=str, help='Path to output image')
    
    args = parser.parse_args()
    
    HoadonOCR().find_label(img=args.input_file)