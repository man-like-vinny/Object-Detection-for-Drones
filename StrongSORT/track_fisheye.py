import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import math


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages, LoadStreams
from yolov7.utils.general import (check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, check_file)
from yolov7.utils.torch_utils import select_device, time_synchronized
from yolov7.utils.plots import plot_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT


VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes


@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        focal_length=150,
        height=0,
        width=0
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
        yolo_weights = Path(yolo_weights[0])
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    save_dir = Path(save_dir)
    (save_dir if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    model = attempt_load(Path(yolo_weights), map_location=device)  # load FP32 model
    names, = model.names,
    stride = model.stride.max().cpu().numpy()  # model stride
    imgsz = check_img_size(imgsz[0], s=stride)  # check image size

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        nr_sources = len(dataset.sources)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                half,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
        strongsort_list[i].model.warmup()
    outputs = [None] * nr_sources
    
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run tracking
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap) in enumerate(dataset):
        s = ''
        t1 = time_synchronized()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_synchronized()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im)
        t3 = time_synchronized()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred[0], conf_thres, iou_thres, classes, agnostic_nms)
        dt[2] += time_synchronized() - t3
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name) + str(i)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

            curr_frames[i] = im0

            txt_path = str(save_dir / txt_file_name)  # im.txt
            print('save_dir', save_dir)
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    #print("names: ", names)
                    #print('c', int(c))
                    print('Current Frame:', frame_idx,'name:', names[int(c)], 'num of det:', n, 'conf', det[:, 4])
                    print('names', names)

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                print('confs', confs)
                print('clss', clss)

                print('this is confs cpu:', confs.cpu())
                print('this is clss cpu:', clss.cpu())

                # pass detections to strongsort
                t4 = time_synchronized()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                
                print('outputs:',outputs[i])

                t5 = time_synchronized()
                dt[3] += t5 - t4

                # draw boxes for visualization
                print('Current Frame:', frame_idx)
                
                print('i counter: ',i)

                print('outputs:',outputs[i])
                print('length of output', len(outputs[i])) # <----- this seems to be the issue!

                example = [1,2,3]
                print('length of example', len(example))

                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
    
                        bboxes = output[0:4]
                        print(bboxes)
                        id = output[4]
                        cls = output[5]
                        #print(cls)

                        print('final j value', j)
                        print('final output values', output)
                        print('final conf values:', conf)

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file

                            adjusted_bbox_left, adjusted_bbox_top, adjusted_bbox_w, adjusted_bbox_h = calculate_output(bbox_left,bbox_top,bbox_w,bbox_h,int(height),int(width),int(focal_length))

                            with open(txt_path + '.txt', 'a') as f:
                                # if(cls == 0):
                                #     cls = 3
                                # elif(cls == 2):
                                #     cls = 4
                                # elif(cls == 4):
                                #     cls = 1
                                # elif(cls == 7):
                                #     cls = 5
                                # elif(cls == 5):
                                #     cls = 7
                                # elif(cls == 1): 
                                #     cls = 9
                                # elif(cls == 3):
                                #     cls = 10
                                # elif(cls == 6):
                                #     cls = 6
                          

                                # f.write(('%g,' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                #                                bbox_top, bbox_w, bbox_h, conf, cls, -1, -1))
                                print('updated frame index:', frame_idx)
                                # f.write(('%g,' * 9 + '%g\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                #                                bbox_top, bbox_w, bbox_h, conf, cls, -1, -1))

                                f.write(('%g,' * 9 + '%g\n') % (frame_idx + 1, id, adjusted_bbox_left,  # MOT format
                                                               adjusted_bbox_top, adjusted_bbox_w, adjusted_bbox_h, conf, cls, -1, -1))

                                # f.write(('%g,' * 9) % (frame_idx + 1, id, bbox_left,  
                                #                                bbox_top, bbox_w, bbox_h, 1, -1, -1))
                                # f.write('-1' + '\n')

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            plot_one_box(bboxes, im0, label=label, color=colors[int(cls)], line_thickness=2)
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                print(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

            else:
                strongsort_list[i].increment_ages()
                print('No detections')

            # Stream results
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, imgsz, imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640,640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/fishdrone-combination', help='save results to project/name')
    parser.add_argument('--name', default='fisheye-150', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--height')
    parser.add_argument('--width')
    parser.add_argument('--focal_length')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))

def calculate_output(top_point,left_point,width,height,image_height,image_width,focal_length):
    # cls_list = []
    # confidence_list = []
    # id_num_list = []
    # top_point_list = []
    # left_point_list = []
    # width_list = []
    # height_list = []

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

    return adjusted_top_point, adjusted_left_point, adjusted_width, adjusted_height

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


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)