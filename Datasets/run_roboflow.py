from roboflow import Roboflow
rf = Roboflow(api_key="msHzVwbT4ShJtVJYixSQ")
project = rf.workspace("msc-project-y4qky").project("visdrone-msc-combined-classes-brfds")
dataset = project.version(1).download("yolov7")

