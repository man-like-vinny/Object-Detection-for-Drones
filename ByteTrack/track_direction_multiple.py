from algorithm.object_detector import YOLOv7
from utils.detections import draw
from utils.detections import calculate_output
from tqdm import tqdm
import numpy as np
import cv2
import os

#This file is used to track multiple folders of sequences

yolov7 = YOLOv7()
yolov7.load('yolov7_visdrone_best-full.pt', classes='data.yaml', device='cpu') # use 'gpu' for CUDA GPU inference

parent_folder = "C:/Users/vinay/Videos/EEE-Stores Backup/MSc Project/easy-yolov7/sequences"

fps = 30
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

for folder_name in os.listdir(parent_folder):
    frames_folder = os.path.join(parent_folder, folder_name)

    if os.path.isdir(frames_folder):
        frame_files = sorted([f for f in os.listdir(frames_folder) if os.path.isfile(os.path.join(frames_folder, f))])

        # Read the first frame to get its dimensions
        first_frame_path = os.path.join(frames_folder, frame_files[0])
        first_frame = cv2.imread(first_frame_path)
        height, width, _ = first_frame.shape

        output = cv2.VideoWriter(f'output_{folder_name}.mp4', fourcc, fps, (width, height))

        print('[+] tracking video...\n')
        pbar = tqdm(total=len(frame_files), unit=' frames', dynamic_ncols=True, position=0, leave=True)
        lines = {}
        arrow_lines = []
        arrow_line_length = 50

        try:
            for frame_num, frame_file in enumerate(frame_files):
                frame_path = os.path.join(frames_folder, frame_file)
                frame = cv2.imread(frame_path)

                detections = yolov7.detect(frame, track=True)
                detected_frame = frame
                    
                for detection in detections:
                    color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
                    
                    if 'id' in detection:
                        detection_id = detection['id']

                        if detection_id not in lines:
                            detection['color'] = color
                            lines[detection_id] = {'points':[], 'arrows':[], 'color':color}
                        else:
                            detection['color'] = lines[detection_id]['color']
                            
                        lines[detection_id]['points'].append(np.array([detection['x'] + detection['width']/2, detection['y'] + detection['height']/2], np.int32))
                        points = lines[detection_id]['points']

                        if len(points) >= 2:
                            arrow_lines = lines[detection_id]['arrows']
                            if len(arrow_lines) > 0:
                                distance = np.linalg.norm(points[-1] - arrow_lines[-1]['end'])
                                if distance >= arrow_line_length:
                                    start = np.rint(arrow_lines[-1]['end'] - ((arrow_lines[-1]['end'] - points[-1])/distance)*10).astype(int)
                                    arrow_lines.append({'start':start, 'end':points[-1]})
                            else:
                                distance = 0
                                arrow_lines.append({'start':points[-2], 'end':points[-1]})
                    
                for line in lines.values():
                    arrow_lines = line['arrows']
                    for arrow_line in arrow_lines:
                        detected_frame = cv2.arrowedLine(detected_frame, arrow_line['start'], arrow_line['end'], line['color'], 2, line_type=cv2.LINE_AA)

                detected_frame = draw(frame, detections)
                output.write(detected_frame)
                output_class, output_confidence, output_id, output_top, output_left, output_width, output_height = calculate_output(frame, detections)
                previous_num_frames = None  # Variable to store the previous value of num_frames

                if frame_num != previous_num_frames:  # Check if num_frames is updated
                    # Update the values of output_id, output_top, output_left, output_width, output_height, output_confidence, and output_class
                    cls_list, confidence_list, id_num_list, top_point_list, left_point_list, width_list, height_list = calculate_output(frame, detections)
                    previous_num_frames = frame_num # Save the new num_frames value

                    for i in range(len(cls_list)):
                        output_id = id_num_list[i]
                        output_top = top_point_list[i]
                        output_left = left_point_list[i]
                        output_width = width_list[i]
                        output_height = height_list[i]
                        output_confidence = confidence_list[i]
                        output_class = cls_list[i]

                        with open(f'{folder_name}.txt', 'a') as f:
                            f.write(('%g,' * 9 + '%g\n') % (frame_num + 1, output_id, output_top, output_left, output_width, output_height, output_confidence, output_class, -1, -1))
                            print(frame_num)
                pbar.update(1)
        except KeyboardInterrupt:
            pass

        pbar.close()
        output.release()
yolov7.unload()