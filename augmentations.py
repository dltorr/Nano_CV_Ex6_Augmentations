import copy

import numpy as np 
from PIL import Image
import utils
import json
from utils import check_results, display_results


def calculate_iou(gt_bbox, pred_bbox):
    """
    calculate iou 
    args:
    - gt_bbox [array]: 1x4 single gt bbox
    - pred_bbox [array]: 1x4 single pred bbox
    returns:
    - iou [float]: iou between 2 bboxes
    - [xmin, ymin, xmax, ymax]
    """
    xmin = np.max([gt_bbox[0], pred_bbox[0]])
    ymin = np.max([gt_bbox[1], pred_bbox[1]])
    xmax = np.min([gt_bbox[2], pred_bbox[2]])
    ymax = np.min([gt_bbox[3], pred_bbox[3]])
    
    intersection = max(0, xmax - xmin) * max(0, ymax - ymin)
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    
    union = gt_area + pred_area - intersection
    return intersection / union, [xmin, ymin, xmax, ymax]


def hflip(img, bboxes):
    """
    horizontal flip of an image and annotations
    args:
    - img [PIL.Image]: original image
    - bboxes [list[list]]: list of bounding boxes
    return:
    - flipped_img [PIL.Image]: horizontally flipped image
    - flipped_bboxes [list[list]]: horizontally flipped bboxes
    """
    # IMPLEMENT THIS FUNCTION
    # Flip the image using PIL image method FLIP_LEFT_RIGHT
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Calculate the width and height of the image
    w, h = img.size
   
    # flip bboxes
    # convert the list into numpy array
    bboxes = np.array(bboxes)
    # Copy to a new variable
    flipped_bboxes = copy.copy(bboxes)
    # Move the boxes y0 and y1 to the other side of the imga, x coordinates stay constant
    flipped_bboxes[:, 1] = w - bboxes[:, 3]
    flipped_bboxes[:, 3] = w - bboxes[:, 1]
   
    return flipped_img, flipped_bboxes


def resize(img, boxes, size):
    """
    resized image and annotations
    args:
    - img [PIL.Image]: original image
    - boxes [list[list]]: list of bounding boxes
    - size [array]: 1x2 array [width, height]
    returns:
    - resized_img [PIL.Image]: resized image
    - resized_boxes [list[list]]: resized bboxes
    """
    # IMPLEMENT THIS FUNCTION
    # Resize the image
    resized_image =img.resize(size)
    # Calculate the new size
    w, h = img.size
    # calculate the ratio per direction
    ratio_width = size[0] /w
    ratio_height = size[1] / h
    
    # resize bboxes
    # convert the list into numpy array
    boxes = np.array(boxes)
    #Copy into the new variable
    resized_boxes = copy.copy(boxes)
    #Resize each square dimension according the ratios
    resized_boxes[:, [0, 2]] = resized_boxes[:, [0, 2]] * ratio_height
    resized_boxes[:, [1, 3]] = resized_boxes[:, [1, 3]] * ratio_width
   
    return resized_image, resized_boxes
 

def random_crop(img, boxes,classes, crop_size, min_area=100):
    """
    random cropping of an image and annotations
    args:
    - img [PIL.Image]: original image
    - boxes [list[list]]: list of bounding boxes
    - crop_size [array]: 1x2 array [width, height]
    - min_area [int]: min area of a bbox to be kept in the crop
    returns:
    - cropped_img [PIL.Image]: resized image
    - cropped_boxes [list[list]]: resized bboxes
    """
    # IMPLEMENT THIS FUNCTION
   # crop coordinates
    w, h = img.size
    x1 = np.random.randint(0, w - crop_size[0])
    y1 = np.random.randint(0, h - crop_size[1])
    x2 = x1 + crop_size[0]
    y2 = y1 + crop_size[1]

    # crop the image
    cropped_image = img.crop((x1, y1, x2 ,y2))
    # initialize cropped boxes and classes
    cropped_boxes = []
    cropped_classes = []
    for bb, cl in zip(boxes, classes):
        iou, inter_coord = calculate_iou(bb, [y1, x1, y2, x2])
        
        # some of the bbox overlap with the crop
        if iou > 0:
            # we need to check the size of the new coord
            area = (inter_coord[3] - inter_coord[1]) * (inter_coord[2] - inter_coord[0])
            
            if area > min_area:
                xmin = inter_coord[1] - x1
                ymin = inter_coord[0] - y1
                xmax = inter_coord[3] - x1
                ymax = inter_coord[2] - y1
                cropped_box = [ymin, xmin, ymax, xmax]
                cropped_boxes.append(cropped_box)
                cropped_classes.append(cl)
   
    
    return cropped_image, cropped_boxes , cropped_classes 

if __name__ == "__main__" :
    file_name = 'segment-12208410199966712301_4480_000_4500_000_with_camera_labels_79.png'
    # Use the image in the example
    path = 'data/images/' + file_name
   # filename = os.path.basename(paths)
    # open the image with python
    img = Image.open(path)
    # open annotations
    # Open json file with ground truth data
    with open('data/ground_truth.json') as f:
        ground_truth = json.load(f)
    # filter annotations and open image
    # Seach for the picture data file
    for i in range(0,len(ground_truth)):
        if ground_truth[i]['filename'] == file_name:
            bboxes = ground_truth[i]['boxes']
            
    # Seach for the picture data file
    for i in range(0,len(ground_truth)):
        if ground_truth[i]['filename'] == file_name:
            classes = ground_truth[i]['classes']  
    
    # fix seed to check results
    flipped_img, flipped_bboxes = hflip(img, bboxes)
    # Check the results
    display_results(img, bboxes, flipped_img, flipped_bboxes)
    check_results(flipped_img, flipped_bboxes, aug_type='hflip')
    
    # Call resize function
    # original size is [1920,1280]
    resized_image, resized_boxes = resize(img, bboxes, size=[640, 640])
    display_results(img, bboxes, resized_image, resized_boxes) 
    check_results(resized_image, resized_boxes, aug_type='resize')

    # check random crop
         
    gt_boxes = [g['boxes'] for g in ground_truth if g['filename'] == file_name][0]
    gt_classes = [g['classes'] for g in ground_truth if g['filename'] == file_name][0]
    
    cropped_image, cropped_boxes, cropped_classes = random_crop(img, gt_boxes, gt_classes, [512, 512], min_area=100)
    display_results(img, bboxes, cropped_image, cropped_boxes)
    check_results(cropped_image, cropped_boxes, aug_type='random_crop', classes=cropped_classes)