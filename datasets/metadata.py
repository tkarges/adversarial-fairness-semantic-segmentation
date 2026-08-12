CITYSCAPES_NUM_CLASSES = 19
CITYSCAPES_IGNORE_INDEX = 255
CITYSCAPES_ROOT = 'datasets/cityscapes'

# Frequencies in descending order!
CITYSCAPES_GROUPS = {
    'frequent': [0, 2, 8, 13, 1, 10],
    'infrequent': [14, 12, 6, 16, 17],
    'medium_freq': [10, 5, 11, 9, 4, 3, 18, 7, 15],
    'hard': [3, 4, 9, 12, 17, 5, 6, 18, 14],
    'easy': [0, 13, 10, 2, 8, 15, 1, 16, 11, 7]
}

CITYSCAPES_LABEL_MAPPING = {
    0: 'road',
    1: 'sidewalk',
    2: 'building',
    3: 'wall',
    4: 'fence',
    5: 'pole',
    6: 'traffic light',
    7: 'traffic sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    11: 'person',
    12: 'rider',
    13: 'car',
    14: 'truck',
    15: 'bus',
    16: 'train',
    17: 'motorcycle',
    18: 'bicycle'
}

CITYSCAPES_TRAIN_CROP = (449, 449)
CITYSCAPES_VAL_CROP = (512, 1024)

ADE_20K_NUM_CLASSES = 150
ADE_20K_IGNORE_INDEX = 255
ADE_20K_ROOT = 'datasets/ade20k'

ADE_20K_GROUPS = {
    'hard': [145, 94, 104, 141, 121, 91, 137, 68, 131, 95, 101, 115, 147, 60, 98, 106, 52, 118, 148, 108, 40, 41, 96, 86, 87, 93, 100, 112, 123, 136, 77, 29, 45, 140, 92, 111, 38, 34, 84, 83, 59, 53, 109, 13, 135, 46, 32, 78, 66, 43, 51, 144, 79, 99, 62, 138, 149, 134, 125, 44, 69, 110, 124, 24, 142, 102, 30, 63, 146, 132, 97, 119, 26, 73, 67],
    'easy': [114, 2, 56, 7, 65, 5, 1, 6, 20, 3, 80, 12, 117, 0, 4, 37, 71, 103, 50, 49, 22, 18, 129, 89, 48, 85, 58, 47, 9, 105, 107, 11, 54, 23, 27, 36, 128, 113, 15, 8, 61, 31, 64, 10, 133, 28, 130, 139, 74, 19, 57, 143, 81, 120, 70, 82, 90, 39, 16, 17, 116, 72, 126, 33, 55, 35, 21, 14, 122, 42, 127, 76, 25, 88, 75]
}

VOC_2012_NUM_CLASSES = 21
VOC_2012_IGNORE_INDEX = 255
VOC_2012_ROOT = 'datasets/voc2012'

VOC_2012_GROUPS = {
    'hard': [9, 11, 2, 18, 16, 20, 4, 5, 12, 14],
    'easy': [6, 0, 1, 19, 8, 7, 17, 13, 3, 10, 15]
}

VOC_2012_LABEL_MAPPING = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "dining table",
    12: "dog",
    13: "horse",
    14: "motor bike",
    15: "person",
    16: "potted plant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tv monitor"
}

def get_dataset_info(dataset: str) -> tuple:
    if dataset == 'ade20k':
        return (ADE_20K_NUM_CLASSES, ADE_20K_IGNORE_INDEX, ADE_20K_ROOT, ADE_20K_GROUPS['hard'], ADE_20K_GROUPS['easy'])
    elif dataset == 'cityscapes':
        return (CITYSCAPES_NUM_CLASSES, CITYSCAPES_IGNORE_INDEX, CITYSCAPES_ROOT, CITYSCAPES_GROUPS['hard'], CITYSCAPES_GROUPS['easy'])
    elif dataset == 'voc2012':
        return (VOC_2012_NUM_CLASSES, VOC_2012_IGNORE_INDEX, VOC_2012_ROOT, VOC_2012_GROUPS['hard'], VOC_2012_GROUPS['easy'])
    else:
        raise ValueError(f'Invalid dataset: {dataset}')
    
def get_label_mapping(dataset: str) -> dict:
    if dataset == 'cityscapes':
        return CITYSCAPES_LABEL_MAPPING
    elif dataset == 'voc2012':
        return VOC_2012_LABEL_MAPPING
    else:
        raise ValueError(f'No label mapping available for dataset {dataset}')
    
def get_policy_split_path(dataset):
    if dataset == "cityscapes":
        return "datasets/splits/cityscapes_weight_split_idx.pt", "datasets/histograms/training_hists.npy"
    else:
        return None