# Class Conversion codes
Two scripts have been provided (Tegra and visDrone2YOLO) to convert the VisDrone annotations into YOLO format such that the models and other dataset management tools can recongise the label maps. **WARNING**: without this step, the training will not result in sub-optimal performance and thus, it is important to convert the labels to YOLO format.

Original VisDrone classes are converted to the following:

 `0: pedestrian`  
 `1: people`  
 `2: bicycle`  
 `3: car`  
 `4: van`  
 `5: truck`  
 `6: tricycle`  
 `7: awning-tricycle`  
 `8: bus`  
 `9: motor`