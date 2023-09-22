import os
import torch
from IPython.display import Image, clear_output

#The code below will execute a focal length of f = 150

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-150.pt" --name "fisheye-augemented-150" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000009_03358_v" --focal_length 150 --height 765 --width 1360
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-150.pt" --name "fisheye-augemented-150" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000073_00600_v" --focal_length 150 --height 1080 --width 1920
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-150.pt" --name "fisheye-augemented-150" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000073_04464_v" --focal_length 150 --height 1080 --width 1920
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-150.pt" --name "fisheye-augemented-150" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000077_00720_v" --focal_length 150 --height 765 --width 1360
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-150.pt" --name "fisheye-augemented-150" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000088_00290_v" --focal_length 150 --height 540 --width 960
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-150.pt" --name "fisheye-augemented-150" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000119_02301_v" --focal_length 150 --height 765 --width 1360
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-150.pt" --name "fisheye-augemented-150" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000120_04775_v" --focal_length 150 --height 765 --width 1360
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-150.pt" --name "fisheye-augemented-150" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000161_00000_v" --focal_length 150 --height 540 --width 960
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-150.pt" --name "fisheye-augemented-150" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000188_00000_v" --focal_length 150 --height 540 --width 960
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-150.pt" --name "fisheye-augemented-150" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000201_00000_v" --focal_length 150 --height 756 --width 1344
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-150.pt" --name "fisheye-augemented-150" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000249_02688_v" --focal_length 150 --height 756 --width 1344
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-150.pt" --name "fisheye-augemented-150" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000297_00000_v" --focal_length 150 --height 1071 --width 1904
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-150.pt" --name "fisheye-augemented-150" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000306_00230_v" --focal_length 150 --height 1071 --width 1904
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-150.pt" --name "fisheye-augemented-150" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000297_02761_v" --focal_length 150 --height 382 --width 680
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-150.pt" --name "fisheye-augemented-150" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000355_00001_v" --focal_length 150 --height 765 --width 1360
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-150.pt" --name "fisheye-augemented-150" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000370_00001_v" --focal_length 150 --height 1530 --width 2720
''')

#The code below will execute a focal length of f = 300

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-300.pt" --name "fisheye-augemented-300" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000009_03358_v" --focal_length 300 --height 765 --width 1360
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-300.pt" --name "fisheye-augemented-300" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000073_00600_v" --focal_length 300 --height 1080 --width 1920
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-300.pt" --name "fisheye-augemented-300" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000073_04464_v" --focal_length 300 --height 1080 --width 1920
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-300.pt" --name "fisheye-augemented-300" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000077_00720_v" --focal_length 300 --height 765 --width 1360
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-300.pt" --name "fisheye-augemented-300" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000088_00290_v" --focal_length 300 --height 540 --width 960
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-300.pt" --name "fisheye-augemented-300" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000119_02301_v" --focal_length 300 --height 765 --width 1360
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-300.pt" --name "fisheye-augemented-300" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000120_04775_v" --focal_length 300 --height 765 --width 1360
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-300.pt" --name "fisheye-augemented-300" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000161_00000_v" --focal_length 300 --height 540 --width 960
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-300.pt" --name "fisheye-augemented-300" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000188_00000_v" --focal_length 300 --height 540 --width 960
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-300.pt" --name "fisheye-augemented-300" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000201_00000_v" --focal_length 300 --height 756 --width 1344
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-300.pt" --name "fisheye-augemented-300" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000249_02688_v" --focal_length 300 --height 756 --width 1344
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-300.pt" --name "fisheye-augemented-300" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000297_00000_v" --focal_length 300 --height 1071 --width 1904
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-300.pt" --name "fisheye-augemented-300" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000297_02761_v" --focal_length 300 --height 1071 --width 1904
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-300.pt" --name "fisheye-augemented-300" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000306_00230_v" --focal_length 300 --height 382 --width 680
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-300.pt" --name "fisheye-augemented-300" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000355_00001_v" --focal_length 300 --height 765 --width 1360
''')

os.system('''
python track_fisheye.py --yolo-weights "visdrone-combined-300.pt" --name "fisheye-augemented-300" --strong-sort-weights osnet_x0_25_msmt17.pt --imgsz 960  --save-vid --save-txt --source "./sequences/uav0000370_00001_v" --focal_length 300 --height 1530 --width 2720
''')