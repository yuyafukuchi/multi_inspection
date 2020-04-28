import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage



def padding(image: np.ndarray, width: int = 50):
    return cv2.copyMakeBorder(image,width,width,width,width,cv2.BORDER_REPLICATE)

def translate(image: np.ndarray, distance: list = [1], h: int = 224, w: int = 224):
    padded = padding(image)
    moved_list = []
    H, W, _ = padded.shape
    for d in distance:
        moved_list.append(padded[((H-h)//2)-d:((H+h)//2)-d,((W-w)//2)-d:((W+w)//2)-d])
        moved_list.append(padded[((H-h)//2)-d:((H+h)//2)-d,((W-w)//2)+d:((W+w)//2)+d])
        moved_list.append(padded[((H-h)//2)+d:((H+h)//2)+d,((W-w)//2)-d:((W+w)//2)-d])
        moved_list.append(padded[((H-h)//2)+d:((H+h)//2)+d,((W-w)//2)+d:((W+w)//2)+d])
    return moved_list


if __name__ == "__main__":

    dataset_path = input('Enter dataset path (q:quit):')
    if dataset_path == 'q':
        exit()
    category_path = input('Enter category path (train/OK (default) or test/OK or test/NG, q:quit):')
    if category_path == 'q':
        exit()
    elif not category_path in ['train/OK', 'test/OK', 'test/NG']:
        print('Default path "train/OK" is selected.')
        category_path = 'train/OK'
    tag = '0'
    last_path = tag+'/'+category_path

    path = dataset_path+'/preprocessed/'+last_path
    image_names = os.listdir(path)

    for name in image_names:
        if not os.path.splitext(name)[-1] in ['.jpg']:
            continue
        image = cv2.imread(path+'/'+name)
        moved_list = translate(image=image)
        for i, im in enumerate(moved_list):
            cv2.imwrite(path+'/'+os.path.splitext(name)[0].split('_')[0]+'_t'+str(i)+'_0.jpg',im)
