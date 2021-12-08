#
# You can modify this files
#

import argparse
import cv2
import matplotlib.pyplot as plt
import pytesseract
from pytesseract import Output

from skimage.filters import threshold_local

class HoadonOCR:

    def __init__(self):
        # Init parameters, load model here
        self.model = None
        self.labels = ['highlands', 'starbucks', 'phuclong', 'others']

    # TODO: implement find label
    def find_label(self, image):
        return None
    
    def read_img(self, src):
        res = cv2.imread(src, cv2.IMREAD_COLOR)
        return res
    
    def show_img(self, image):
        cv2.imshow('Result', image)
        cv2.waitKey(0)
        
    def resize_img(self, image, ratio):
        height = int(image.shape[0] * ratio)
        width = int(image.shape[1] * ratio)
        dim = (width, height)
        res = cv2.resize(image, dim, interpolation= cv2.INTER_AREA)
        return res
    
    def crop_image_to_half(self, image):
        cropped_image = image[0 : image.shape[0] // 2]
        return cropped_image
    
    def plot_rgb(self, image):
        plt.figure(figsize=(16, 10))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
    
    def plot_gray(self, image):
        plt.figure(figsize=(16, 10))
        plt.imshow(image, cmap='Greys_r')
        plt.show()
        
    # Detect white region
    def detect_white_region(self, erosion):
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
        rectKernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 20))
        dilated = cv2.dilate(erosion, rectKernel)
                
        opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, rectKernel2)
        closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, rectKernel2)
        
        return (dilated, opening, closing)
            
    # Find edge
    def find_edge(self, closing):
        (thresh, blackAndWhite) = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.plot_gray(blackAndWhite)
        edge = cv2.Canny(blackAndWhite, 30, 30, apertureSize=3)
        self.plot_gray(edge)
        return (blackAndWhite, edge)
    
    # Detect all contours
    def find_contours(self, image, blackAndWhite):
        contours, hierachy = cv2.findContours(blackAndWhite, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 3)
        self.plot_rgb(image)
        return (image_with_contours, contours)
    
    # Get largest contours
    def get_largest_contours(self, image, contours):
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        image_with_largest_contours = cv2.drawContours(image.copy(), largest_contours, -1, (0,255,0), 3)
        return (image_with_largest_contours, largest_contours)
                
    # Approximate the contour by a more primitive polygon shape
    def approximate_contour(self, contour):
        peri = cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, 0.032 * peri, True)
    
    # Get receipt contour
    def get_receipt_contour(self, contours):
        for c in contours:
            approx = self.approximate_contour(c)
            
            if len(approx) == 4:
                return approx    
            
    def tesseract(self, image):
        d = pytesseract.image_to_data(image, output_type=Output.DICT)
        n_boxes = len(d['level'])
        boxes = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        for i in range(n_boxes):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])    
            boxes = cv2.rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
        self.plot_rgb(boxes)
        
    # Using template matching for logo
    def match_logo(self, image, template):
        clone = image.copy()
        H, W = template.shape
        result = cv2.matchTemplate(clone, template, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result);
        location = max_loc
        bottom_right = (location[0] + W, location[1] + H)
        cv2.rectangle(clone, location, bottom_right, 255, 5)
        cv2.imshow('Res', clone)
        cv2.waitKey(0)
                
    # Detect logo
    def detect_logo(self, image):
        for i in range(2):
            temp = cv2.imread()
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', type=str, help='Path to input image')
    parser.add_argument('--output_file', '-o', type=str, help='Path to output image')
    
    args = parser.parse_args()
    
    bill = HoadonOCR()
    
    img = bill.read_img(src=args.input_file)
    
    ratio = 500 / img.shape[0]
    original = img.copy()
    img = bill.resize_img(img, ratio)
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('./templates/template_starbucks_1.png')
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # cleimg = clahe.apply(gray)
    # # bill.plot_gray(cleimg)
    # blurred = cv2.GaussianBlur(cleimg, (5, 5), 1)
    # # bill.plot_gray(blurred)
    # blurred = cv2.medianBlur(blurred, 7)
   
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    # erosion = cv2.erode(blurred, kernel, iterations = 1)
    # (_, _, closing) = bill.detect_white_region(erosion)
    # (blackAndWhite, edge) = bill.find_edge(closing)
    # bill.find_contours(img, blackAndWhite)
    bill.match_logo(gray, template)