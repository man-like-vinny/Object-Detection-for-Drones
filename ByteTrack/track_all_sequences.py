import os
import torch
from IPython.display import Image, clear_output

#Code to execute the finedtuned standard lens model on a standard dataset
os.system('''
python track_direction.py --weights "yolov7_visdrone_combination.pt" --source "uav0000009_03358_v" && python track_direction.py --weights "yolov7_visdrone_combination.pt" --source "uav0000073_00600_v" && python track_direction.py --weights "yolov7_visdrone_combination.pt" --source "uav0000073_04464_v" && python track_direction.py --weights "yolov7_visdrone_combination.pt" --source "uav0000077_00720_v" && python track_direction.py --weights "yolov7_visdrone_combination.pt" --source "uav0000088_00290_v" && python track_direction.py --weights "yolov7_visdrone_combination.pt" --source "uav0000119_02301_v" && python track_direction.py --weights "yolov7_visdrone_combination.pt" --source "uav0000120_04775_v" && python track_direction.py --weights "yolov7_visdrone_combination.pt" --source "uav0000161_00000_v" && python track_direction.py --weights "yolov7_visdrone_combination.pt" --source "uav0000188_00000_v" && python track_direction.py --weights "yolov7_visdrone_combination.pt" --source "uav0000201_00000_v" && python track_direction.py --weights "yolov7_visdrone_combination.pt" --source "uav0000249_02688_v" && python track_direction.py --weights "yolov7_visdrone_combination.pt" --source "uav0000297_00000_v" && python track_direction.py --weights "yolov7_visdrone_combination.pt" --source "uav0000297_02761_v" && python track_direction.py --weights "yolov7_visdrone_combination.pt" --source "uav0000306_00230_v" && python track_direction.py --weights "yolov7_visdrone_combination.pt" --source "uav0000355_00001_v" && python track_direction.py --weights "yolov7_visdrone_combination.pt" --source "uav0000370_00001_v"
''')

#Code to execute the finedtuned fisheye lens model on a fisheye augmented dataset
os.system('''
python track_direction.py --weights "visdrone-combined-150.pt" --source "fisheye-150/uav0000009_03358_v" && python track_direction.py --weights "visdrone-combined-150.pt" --source "fisheye-150/uav0000073_00600_v" && python track_direction.py --weights "visdrone-combined-150.pt" --source "fisheye-150/uav0000073_04464_v" && python track_direction.py --weights "visdrone-combined-150.pt" --source "fisheye-150/uav0000077_00720_v" && python track_direction.py --weights "visdrone-combined-150.pt" --source "fisheye-150/uav0000088_00290_v" && python track_direction.py --weights "visdrone-combined-150.pt" --source "fisheye-150/uav0000119_02301_v" && python track_direction.py --weights "visdrone-combined-150.pt" --source "fisheye-150/uav0000120_04775_v" && python track_direction.py --weights "visdrone-combined-150.pt" --source "fisheye-150/uav0000161_00000_v" && python track_direction.py --weights "visdrone-combined-150.pt" --source "fisheye-150/uav0000188_00000_v" && python track_direction.py --weights "visdrone-combined-150.pt" --source "fisheye-150/uav0000201_00000_v" && python track_direction.py --weights "visdrone-combined-150.pt" --source "fisheye-150/uav0000249_02688_v" && python track_direction.py --weights "visdrone-combined-150.pt" --source "fisheye-150/uav0000297_00000_v" && python track_direction.py --weights "visdrone-combined-150.pt" --source "fisheye-150/uav0000297_02761_v" && python track_direction.py --weights "visdrone-combined-150.pt" --source "fisheye-150/uav0000306_00230_v" && python track_direction.py --weights "visdrone-combined-150.pt" --source "fisheye-150/uav0000355_00001_v" && python track_direction.py --weights "visdrone-combined-150.pt" --source "fisheye-150/uav0000370_00001_v"
''')

#Code to execute the finedtuned fisheye lens model on a standard dataset
os.system('''
python track_direction.py --weights "visdrone-combined-150.pt" --source "uav0000009_03358_v" 
''')

os.system('''
python track_direction.py --weights "visdrone-combined-150.pt" --source "uav0000073_00600_v" 
''')

os.system('''
python track_direction.py --weights "visdrone-combined-150.pt" --source "uav0000073_04464_v" 
''')

os.system('''
python track_direction.py --weights "visdrone-combined-150.pt" --source "uav0000077_00720_v" 
''')

os.system('''
python track_direction.py --weights "visdrone-combined-150.pt" --source "uav0000088_00290_v" 
''')

os.system('''
python track_direction.py --weights "visdrone-combined-150.pt" --source "uav0000119_02301_v" 
''')

os.system('''
python track_direction.py --weights "visdrone-combined-150.pt" --source "uav0000120_04775_v" 
''')

os.system('''
python track_direction.py --weights "visdrone-combined-150.pt" --source "uav0000161_00000_v" 
''')

os.system('''
python track_direction.py --weights "visdrone-combined-150.pt" --source "uav0000188_00000_v" 
''')

os.system('''
python track_direction.py --weights "visdrone-combined-150.pt" --source "uav0000201_00000_v" 
''')

os.system('''
python track_direction.py --weights "visdrone-combined-150.pt" --source "uav0000249_02688_v" 
''')

os.system('''
python track_direction.py --weights "visdrone-combined-150.pt" --source "uav0000297_00000_v" 
''')

os.system('''
python track_direction.py --weights "visdrone-combined-150.pt" --source "uav0000297_02761_v" 
''')

os.system('''
python track_direction.py --weights "visdrone-combined-150.pt" --source "uav0000306_00230_v" 
''')

os.system('''
python track_direction.py --weights "visdrone-combined-150.pt" --source "uav0000355_00001_v" 
''')

os.system('''
python track_direction.py --weights "visdrone-combined-150.pt" --source "uav0000370_00001_v" 
''')

#-------------------------------------------------------------------------------------------
os.system('''
python track_direction.py --weights "visdrone-combined-300.pt" --source "uav0000009_03358_v"  
''')

os.system('''
python track_direction.py --weights "visdrone-combined-300.pt" --source "uav0000073_00600_v"  
''')

os.system('''
python track_direction.py --weights "visdrone-combined-300.pt" --source "uav0000073_04464_v"  
''')

os.system('''
python track_direction.py --weights "visdrone-combined-300.pt" --source "uav0000077_00720_v"  
''')

os.system('''
python track_direction.py --weights "visdrone-combined-300.pt" --source "uav0000088_00290_v"  
''')

os.system('''
python track_direction.py --weights "visdrone-combined-300.pt" --source "uav0000119_02301_v"  
''')

os.system('''
python track_direction.py --weights "visdrone-combined-300.pt" --source "uav0000120_04775_v"  
''')

os.system('''
python track_direction.py --weights "visdrone-combined-300.pt" --source "uav0000161_00000_v"  
''')

os.system('''
python track_direction.py --weights "visdrone-combined-300.pt" --source "uav0000188_00000_v"  
''')

os.system('''
python track_direction.py --weights "visdrone-combined-300.pt" --source "uav0000201_00000_v"  
''')

os.system('''
python track_direction.py --weights "visdrone-combined-300.pt" --source "uav0000249_02688_v"  
''')

os.system('''
python track_direction.py --weights "visdrone-combined-300.pt" --source "uav0000297_00000_v"  
''')

os.system('''
python track_direction.py --weights "visdrone-combined-300.pt" --source "uav0000297_02761_v"  
''')

os.system('''
python track_direction.py --weights "visdrone-combined-300.pt" --source "uav0000306_00230_v"  
''')

os.system('''
python track_direction.py --weights "visdrone-combined-300.pt" --source "uav0000355_00001_v"  
''')

os.system('''
python track_direction.py --weights "visdrone-combined-300.pt" --source "uav0000370_00001_v"  
''')