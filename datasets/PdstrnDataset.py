import re
from pathlib import Path

def get_annotations(directory):
    """
    Get the annotations from the directory.
    """
    annotations = []
    for file in Path(directory).rglob('*.txt'):
        data = parce_pscl_annotations(file)
        if data:
            annotations.append(data)
    return annotations

def parce_pscl_annotations(file):
    """
    Parse the annotations from the file and add them to the annotations list.
    """
    with open(file, 'r') as f:
        lines = f.readlines()

    data ={
        'image_path': None,
        'mask_path': None,
        'image_size': None,
        'boxes': [],
        'labels': [],
        'masks': []
    }

    image_path_rgx = re.compile(r'^Image filename\s*:\s*"(.+)"')
    bbox_rgx = re.compile(r'Bounding box.*?:\s+\((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)')
    mask_rgx = re.compile(r'Pixel mask.*?:\s+"(.+)"')
    
    object_index = 1

    for line in lines:
        img = image_path_rgx.search(line)
        if img:
            data['image_path'] = img.group(1)
        
        bbox = bbox_rgx.search(line)
        if bbox:
            x1, y1, x2, y2 = map(int, bbox.groups())
            data['boxes'].append([x1, y1, x2, y2])
            data['labels'].append(object_index)
            object_index += 1
        
        mask = mask_rgx.search(line)
        if mask:
            data['masks'].append(mask.group(1))
        
    
    return data

data = get_annotations('./database/PennFudanPed/Annotation')

print(data)