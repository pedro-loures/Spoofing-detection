# Data processing
import utils as ut
import json
import os 

# Training and validation
from sklearn.model_selection import train_test_split

# Pre-processing
from PIL import Image

annotations_file = open(os.path.join(ut.DATA_FOLDER, 'annotations.json'))
annotations_dict = json.load(annotations_file)
keys = list(annotations_dict.keys())
labels = [annotations_dict[key][43] for key in keys]


X_train, X_test, y_train, y_test = train_test_split(keys, labels, test_size=0.33, random_state=42)

# Evaluate percentage of classes in both train and test

sum(y_train)/len(y_train), sum(y_test)/len(y_test)

def resize_image(image_path, output_path, new_size):
    with Image.open(image_path) as image:
        resized_image = image#.resize(new_size)
        resized_image.save(output_path)


new_size = (244, 244)
for idx, key in enumerate(X_test):
    image_name = 'image_' + str(idx) + '.jpg'
    if y_test[idx]: # If Spoof
        path = 'data\\test\\spoof'

    else:
        path = 'data\\test\\live'  
    
    original_path = os.path.join(ut.DATA_FOLDER, key)
    output_path = os.path.join(path, image_name) 
    resize_image(original_path, output_path, new_size)




for idx, key in enumerate(X_train):
    image_name = 'image_' + str(idx) + '.jpg'
    if y_train[idx]: # If Spoof
        path = 'data\\train\\spoof'

    else:
        path = 'data\\train\\live'  
    
    original_path = os.path.join(ut.DATA_FOLDER, key)
    output_path = os.path.join(path, image_name) 
    resize_image(original_path, output_path, new_size)



