import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def align_diff(base_image: np.ndarray, search_image: np.ndarray, size: int = 224, top=None, left=None):
    # 基準画像の縦横を取得
    h, w, _= base_image.shape
    # size, top, leftをチェック
    if size <= min(h,w):
        if top == None:
            top = (h-size)//2
        elif top+size > h:
            print(f'Error: invalid value (top: {top})')
            return
        if left == None:
            left = (w-size)//2
        elif left+size > w:
            print(f'Error: invalid value (left: {left})')
            return
    else:
        print(f'Error: size {size} is bigger than image size ({h},{w})')
        return
    # 基準画像の切り取り
    base_crop = base_image[top:top+size, left:left+size]  
    # 探索範囲を決定
    search_size = min(int(size*1.5), h, w) # 平行移動の探索範囲を調整したいときはここを変更する
    search_top = top+(size-search_size)//2
    if search_size == h or search_top<0:
        search_top = 0
    search_left = left+(size-search_size)//2
    if search_size == w or search_left<0:
        search_left = 0
    # 探索画像の切り取り
    search_crop = search_image[search_top:search_top+search_size, search_left:search_left+search_size]
    # 位置合わせ
    max_vals = []
    max_locs = []
    center = (search_size//2, search_size//2) # 回転中心
    angles = np.arange(-5., 5., 0.5).tolist() # 角度の探索範囲を調整したいときはここを変更する
    for angle in angles:
        trans = cv2.getRotationMatrix2D(center, angle , 1)
        rot_img = cv2.warpAffine(search_crop, trans, (search_size, search_size))
        res = cv2.matchTemplate(rot_img, base_crop, cv2.TM_CCORR_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        max_vals.append(max_val)
        max_locs.append(max_loc)
    max_idx = np.argmax(np.array(max_vals))
    max_angle = angles[max_idx]
    max_loc = max_locs[max_idx]
    # 画像の回転と切り出し
    trans = cv2.getRotationMatrix2D(center, max_angle , 1)
    result = cv2.warpAffine(search_crop, trans, (search_size, search_size))[max_loc[1]:max_loc[1]+size, max_loc[0]:max_loc[0]+size]
    # 差分画像の出力
    diff = cv2.absdiff(base_crop, result)
    return diff

def normalize(image: np.ndarray):
    return cv2.normalize(image,None,0,255,cv2.NORM_MINMAX)

def resize(image: np.ndarray, size: tuple = (245,205)):
    return cv2.resize(image, size)

def crop(image: np.ndarray, h: int = 2000, w: int = 2000):
    H, W, _ = image.shape
    return image[(H-h)//2:(H+h)//2,(W-w)//2:(W+w)//2]

def resize_normalize(image: np.ndarray): # not used
    return normalize(resize(image))

def crop_resize_normalize(image: np.ndarray):
    return normalize(resize(crop(image),(224,224)))

def padding(image: np.ndarray, width: int = 50):
    return cv2.copyMakeBorder(image,width,width,width,width,cv2.BORDER_REPLICATE)

def resize_padding_align_diff_normalize(before: np.ndarray, after: np.ndarray):
    before = padding(resize(before))
    after = padding(resize(after))
    diff = align_diff(before, after, 224)
    return normalize(diff)

def crop_resize_padding_align_diff_normalize(before: np.ndarray, after: np.ndarray):
    before = padding(resize(crop(before, 2000, 2200), (240, 240)))
    after = padding(resize(crop(after, 2000, 2200), (240, 240)))
    diff = align_diff(before, after, 224)
    return normalize(diff)

def diff_dir(base_dir: str, search_dir: str, save_dir: str):
    base_paths=os.listdir(base_dir)
    for filename in base_paths:
        print(filename)
        if os.path.splitext(filename)[-1] not in ['.jpg']:
            continue
        before = cv2.imread(base_dir+'/'+filename)
        after = cv2.imread(search_dir+'/'+filename)
        # diff = resize_padding_align_diff_normalize(before, after)
        diff = crop_resize_padding_align_diff_normalize(before, after)
        cv2.imwrite(save_dir+'/'+filename,diff)

def normalize_dir(base_dir: str, save_dir: str):
    base_paths=os.listdir(base_dir)
    for filename in base_paths:
        if os.path.splitext(filename)[-1] not in ['.jpg']:
            continue
        image = cv2.imread(base_dir+'/'+filename)
        image = crop_resize_normalize(image)
        cv2.imwrite(save_dir+'/'+filename,image)
    

if __name__ == "__main__":

    mode = input('Enter process mode (FILE or DIR):')

    if mode == 'FILE':
        method = input('align_diff or normalize? (a or n):')
        if method == 'a':
            before = cv2.imread(input('Enter PLAIN image path:'))
            after = cv2.imread(input('Enter PRINTED image path:'))
            save_path = input('Enter save path:')
            diff = resize_padding_align_diff_normalize(before, after)
            cv2.imwrite(save_path+'/diff.jpg', diff)
        elif method == 'n':
            image = cv2.imread(input('Enter image path:'))
            image = crop_resize_normalize(image)
            cv2.imwrite(input('Enter save path:')+'/'+filename,image)
        else:
            print('Error: Enter the correct method (a or e).')


    elif mode == 'DIR':
        method = input('align_diff or normalize? (a or n):')
        dataset_path = input('Enter dataset path (q:quit):')
        if dataset_path == 'q':
            exit()
        category_path = input('Enter category path (train/OK (default) or test/OK or test/NG, q:quit):')
        if category_path == 'q':
            exit()
        elif not category_path in ['train/OK', 'test/OK', 'test/NG']:
            print('Default path "train/OK" is selected.')
            category_path = 'train/OK'

        tag = input('Enter tag: ')

        last_path = tag+'/'+category_path

        if not os.path.exists(dataset_path+'/preprocessed/'+last_path):
            os.makedirs(dataset_path+'/preprocessed/'+last_path)

        if method == 'a':
            diff_dir(base_dir=dataset_path+'/original_plain/'+last_path,
                    search_dir=dataset_path+'/original_printed/'+last_path,
                    save_dir=dataset_path+'/preprocessed/'+last_path)
        elif method == 'n':
            normalize_dir(base_dir=dataset_path+'/original_printed/'+last_path,
                        save_dir=dataset_path+'/preprocessed/'+last_path)
        else:
            print('Error: Enter the correct method (a or n).')