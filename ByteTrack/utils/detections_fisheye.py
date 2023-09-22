from collections import OrderedDict
from PIL import ImageColor
import numpy as np
import json
import cv2
import math


class Point:
    def __init__(self, raw_point):
        self.x = raw_point[0]
        self.y = raw_point[1]

    def to_string(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ')'
    
    def to_dict(self):
        return {'x':self.x, 'y':self.y}


class Box:
    def __init__(self, class_name, confidence, raw_corner_points, color, track_id=None):
        self.class_name = class_name
        self.confidence = confidence
        self.raw_corner_points = raw_corner_points
        self.top_left_point = Point(raw_corner_points[0])
        self.bottom_right_point = Point(raw_corner_points[1])
        self.width =  self.bottom_right_point.x - self.top_left_point.x
        self.height = self.bottom_right_point.y - self.top_left_point.y
        self.color = color
        self.track_id = track_id

    def to_dict(self):
        box = OrderedDict([
            ('class', self.class_name),
            ('confidence', self.confidence),
            ('x', self.top_left_point.x),
            ('y', self.top_left_point.y),
            ('width', self.width),
            ('height', self.height),
            ('color', self.color)
        ])
        if self.track_id is not None:
            box['id'] = self.track_id
        return box


class Detections:
    def __init__(self, raw_detection, classes, tracking=False):
        self.__raw_detection = raw_detection
        self.__classes = classes
        self.__boxes = []
        self.__tracking = tracking
        self.__point1_index = 0
        self.__point2_index = 1
        self.__point3_index = 2
        self.__point4_index = 3
        self.__tracking_index = 4
        self.__class_index = 5 if tracking else 5
        self.__confidence_index = 6 if tracking else 4
        self.__extract_boxes()

    def __extract_boxes(self):
         for raw_box in self.__raw_detection:
            track_id = None
            if self.__tracking:
                track_id = int(raw_box[self.__tracking_index])
            class_id = int(raw_box[self.__class_index])
            raw_corner_points = (int(raw_box[self.__point1_index]), int(raw_box[self.__point2_index])), (int(raw_box[self.__point3_index]), int(raw_box[self.__point4_index]))
            confidence = raw_box[self.__confidence_index]
            dataset_class = self.__classes[class_id]
            class_name = dataset_class['name']
            class_color = dataset_class['color']
            box = Box(class_name, confidence, raw_corner_points, class_color, track_id=track_id)
            self.__boxes.append(box)
        
    def get_boxes(self):
        return self.__boxes

    def to_dict(self):
        boxes = []
        for box in self.__boxes:
            boxes.append(box.to_dict())
        return boxes

    def to_json(self):
        boxes = self.to_dict()
        return json.dumps(boxes, indent=4)


def plot_box(image, top_left_point, bottom_right_point, width, height, label, color=(210,240,0), padding=6, font_scale=0.35):
    label = label.upper()
    
    cv2.rectangle(image, (top_left_point['x'] - 1, top_left_point['y']), (bottom_right_point['x'], bottom_right_point['y']), color, thickness=2, lineType=cv2.LINE_AA)
    res_scale = (image.shape[0] + image.shape[1])/1600
    font_scale = font_scale * res_scale
    font_width, font_height = 0, 0
    font_face = cv2.FONT_HERSHEY_DUPLEX
    text_size = cv2.getTextSize(label, font_face, fontScale=font_scale, thickness=1)[0]

    if text_size[0] > font_width:
        font_width = text_size[0]
    if text_size[1] > font_height:
        font_height = text_size[1]
    if top_left_point['x'] - 1 < 0:
        top_left_point['x'] = 1
    if top_left_point['x'] + font_width + padding*2 > image.shape[1]:
        top_left_point['x'] = image.shape[1] - font_width - padding*2
    if top_left_point['y'] - font_height - padding*2  < 0:
        top_left_point['y'] = font_height + padding*2
    
    p3 = top_left_point['x'] + font_width + padding*2, top_left_point['y'] - font_height - padding*2
    cv2.rectangle(image, (top_left_point['x'] - 2, top_left_point['y']), p3, color, -1, lineType=cv2.LINE_AA)
    x = top_left_point['x'] + padding
    y = top_left_point['y'] - padding
    cv2.putText(image, label, (x, y), font_face, font_scale, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)
    return image

def draw(image, detections):
    image_copy = image.copy()
    for box in detections:
        class_name = box['class']
        conf = box['confidence']
        text = ''
        if 'text' in box:
            text = box['text']
            if len(text) > 50:
                text = text[:50] + ' ...'
        label = (str(box['id']) + '. ' if 'id' in box else '') + class_name + ' ' + str(int(conf*100)) + '%' + ((' | ' + text) if ('text' in box and len(box['text']) > 0 and not box['text'].isspace()) else '')
        width = box['width']
        height = box['height']
        color = box['color']

        if isinstance(color, str):
            color = ImageColor.getrgb(color)
            color = (color[2], color[1], color[0])
        
        top_left_point = {'x':box['x'], 'y':box['y']}
        bottom_right_point = {'x':box['x'] + width, 'y':box['y'] + height}
        image_copy = plot_box(image_copy, top_left_point, bottom_right_point, width, height, label, color=color)
    return image_copy

def calculate_output(image, detections,image_height,image_width,focal_length):
    cls_list = []
    confidence_list = []
    id_num_list = []
    top_point_list = []
    left_point_list = []
    width_list = []
    height_list = []
    #awning_counter = 0
    #person_counter = 0
    
    for box in detections:

        cls = 0

        print('box', box)
        class_name = box['class']
        if(class_name == 'bicycle'):
            cls = 3
        elif(class_name == 'car'):
            cls = 4
        elif(class_name == 'pedestrian'):
            cls = 1
        elif(class_name == 'van'):
            cls = 5
        elif(class_name == 'tricycle'):
            cls = 7
        elif(class_name == 'bus'): 
            cls = 9
        elif(class_name == 'awning-tricycle'):
            cls = 7
        elif(class_name == 'people'):
            cls = 1
        elif(class_name == 'motor'):
            cls = 10 
        elif(class_name == 'truck'):
            cls = 6       

        # if(class_name == 'bicycle'):
        #     cls = 3
        # elif(class_name == 'car'):
        #     cls = 4
        # elif(class_name == 'pedestrian'):
        #     cls = 1
        # elif(class_name == 'people'):
        #     cls = 1
        #     #person_counter+=1
        # elif(class_name == 'van'):
        #     cls = 5
        # elif(class_name == 'tricycle'):
        #     cls = 7
        # elif(class_name == 'awning-tricycle'):
        #     cls = 7
        #     #awning_counter+=1
        # elif(class_name == 'bus'): 
        #     cls = 9
        # elif(class_name == 'motor'):
        #     cls = 10 
        # elif(class_name == 'truck'):
        #     cls = 6 

        # if(class_name == '3'):
        #     cls = 3
        # elif(class_name == '4'):
        #     cls = 4
        # elif(class_name == '1'):
        #     cls = 1
        # elif(class_name == '5'):
        #     cls = 5
        # elif(class_name == '7'):
        #     cls = 7
        # elif(class_name == '9'): 
        #     cls = 9
        # elif(class_name == '10'):
        #     cls = 10 
        # elif(class_name == '6'):
        #     cls = 6 
        
        #class_name = box['class']
        conf = box['confidence']
        confidence = float(conf)
        id_num = int(box['id'])
        width = box['width']
        height = box['height']
        top_point = int(box['x'])
        left_point = int(box['y'])

        print("value of cls", cls)

        #distorting images:

        u_x = int(image_width/2)
        u_y = int(image_height/2)
        if image_width/2 % 2 == 0:
            u_x -= 1 
        if image_height/2 % 2 == 0:
            u_y -= 1
        
        _, top_border = distort_coordinates(u_x, 0, u_x, u_y, int(focal_length)) # top_middle = (u_x, 0), dist_tm
        left_border, _ = distort_coordinates(0, u_y, u_x, u_y, int(focal_length)) # middle_left = (0, u_y), dist_ml
        adjusted_top_point, adjusted_left_point, adjusted_width, adjusted_height = distort_labels(top_point,left_point,width,height, int(focal_length), u_x, u_y,crop=True, top_border=top_border, left_border=left_border)

        cls_list.append(cls)
        confidence_list.append(confidence)
        id_num_list.append(id_num)
        # top_point_list.append(top_point)
        # left_point_list.append(left_point)
        # width_list.append(width)
        # height_list.append(height)

        top_point_list.append(adjusted_top_point)
        left_point_list.append(adjusted_left_point)
        width_list.append(adjusted_width)
        height_list.append(adjusted_height)

        #print('box', box)
        
    return cls_list, confidence_list, id_num_list, top_point_list, left_point_list, width_list, height_list

def distort_coordinates(x, y, u_x, u_y, f):
    norm_y = y-u_y
    norm_x = x-u_x

    # Get gradient of line going through x,y
    m = norm_y if norm_x==0 else (norm_y)/(norm_x)
            
    # Formula: rf = f.arctan(rc/f)
    # Calculate hypotenuse length of pixel from conventional image
    rc = math.sqrt(pow(norm_x,2) + pow(norm_y,2))
    # print("Calculated rc: ", rc)
            
    numerator = pow((f*(math.atan(rc/f))),2)
    # print("Calculated numerator: ", numerator)
    denominator = 1+pow(m,2)
    # print("Calculated denominator: ", denominator)
    x_f = math.sqrt(numerator/denominator)

    if norm_x < 0: # If x is originally negative, make x_f negative
        x_f = -x_f
    # print("Calculated x_f: ", x_f)
    y_f = x_f * m
    # print("Calculated y_f: ", y_f)

    # Resulting points are relative to u_x and u_y
    # print("Mapping : ", x, ", ", y, " : ", x_f+u_x, ", ", y_f+u_y)

    # Return the distorted pair of coordinates
    return int(x_f + u_x), int(y_f + u_y)

def distort_labels(left,top,width,height, f, u_x, u_y, crop=False, top_border=0, left_border=0):
    # Create a new file, return an error if it already exists
    #original_file = open(label_file)
    #parent_dir = label_file.parents[1]
    # print(str(label_file.parent).split("\\")[1])
    #result_file = open(str(parent_dir) + "/fisheye/" + focal_length + "/labels/" + label_file.stem+".txt", "w")

    u_x_crop = u_x
    u_y_crop = u_y

    if crop:
        u_x_crop = u_x - left_border
        u_y_crop = u_y - top_border


        # inst_class, x_centre, y_centre, bbox_w, bbox_h = map(float, line.split(" "))
        # x_centre = x_centre * u_x*2
        # bbox_w = int(bbox_w * u_x*2)
        # y_centre = y_centre * u_y*2
        # bbox_h = int(bbox_h * u_y*2)
        
        # values = list(map(float, line.split(",")))
        # frame_id = values[0]
        # class_id = values[1]
        # conf = values[6]
        # second_last = values[8]
        # last = values[9]

        #inst_class = values[7]  # 8th column as inst_class
    x_centre = left * u_x * 2  # 3rd column as x_centre
    y_centre = top * u_y * 2  # 4th column as y_centre
    bbox_w = int(width * u_x * 2)  # 5th column as bbox_w
    bbox_h = int(height * u_y * 2)  # 6th column as bbox_h

    bbox_l = int(x_centre - bbox_w/2)
    bbox_t = int(y_centre - bbox_h/2)

    dist_tl = distort_coordinates(bbox_l, bbox_t, u_x, u_y, f)
    dist_bl = distort_coordinates(bbox_l, bbox_t+bbox_h, u_x, u_y, f)
    dist_br = distort_coordinates(bbox_l+bbox_w, bbox_t+bbox_h, u_x, u_y, f)
    dist_tr = distort_coordinates(bbox_l+bbox_w, bbox_t, u_x, u_y, f)

        # Calculate new metrics from the corner distorted corner coordinates
    bbox_l = min(dist_tl[0], dist_bl[0]) # Left-most x coordinate
    bbox_t = min(dist_tl[1], dist_tr[1]) # Top-most y coordinate
    bbox_w = max(dist_br[0], dist_tr[0]) - bbox_l
    bbox_h = max(dist_br[1], dist_bl[1]) - bbox_t

    if crop: # Determine 
        bbox_l -= left_border
        bbox_t -= top_border

        # Convert back to format for label file
    x_centre = (bbox_l + bbox_w/2)/(u_x_crop*2)
    y_centre = (bbox_t + bbox_h/2)/(u_y_crop*2)

        #ret_items = [inst_class, x_centre, y_centre, bbox_w/(u_x_crop*2), bbox_h/(u_y_crop*2)]

        #ret_items = [int(frame_id), int(class_id), x_centre, y_centre, bbox_w/(u_x_crop*2), bbox_h/(u_y_crop*2), int(conf), int(inst_class), int(second_last), int(last)]
        #ret_items = [int(frame_id), int(class_id), round(x_centre,3), round(y_centre,3), round(bbox_w/(u_x_crop*2),3), round(bbox_h/(u_y_crop*2),3), conf, int(inst_class), int(second_last), int(last)]

        #result_file.write(",".join([str(n) for n in ret_items])+"\n")
    # Return the filepath that the annotations have been written to
    return round(x_centre,3), round(y_centre,3), round(bbox_w/(u_x_crop*2),3), round(bbox_h/(u_y_crop*2),3)