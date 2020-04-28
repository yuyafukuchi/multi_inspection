# -*- Coding: utf-8 -*-
import cv2
import numpy as np
import os

class Mask:

    def main(self, input_image):
        # HSVに変換
        hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

        # ## for debug
        # cv2.imshow('hsv', hsv[:,:,0])
        # cv2.waitKey(0)

        # 指定した色相以外のピクセルをマスク
        mask_hue = cv2.bitwise_not(self.__hue_mask(input_image=hsv, thresh_lower=30, thresh_upper=180))

        # モロフォルジー演算
        kernel = np.ones((3, 3), np.uint8)
        mask_hue = cv2.dilate(mask_hue, kernel, iterations=1)
        mask_hue = cv2.erode(mask_hue, kernel, iterations=2)
        mask_hue = cv2.dilate(mask_hue, kernel, iterations=1)
        mask_hue = cv2.erode(mask_hue, kernel, iterations=2)
        mask_hue = cv2.dilate(mask_hue, kernel, iterations=1)

        # ## for debug
        # cv2.imshow('mask_hue', mask_hue)
        # cv2.waitKey(0)
        #

        # 指定した彩度のピクセルをマスク
        mask_sat = cv2.bitwise_not(self.__sat_mask(input_image=hsv, thresh_lower=0, thresh_upper=50))

        # モロフォルジー演算
        kernel = np.ones((3, 3), np.uint8)
        mask_sat = cv2.erode(mask_sat, kernel, iterations=1)
        mask_sat = cv2.dilate(mask_sat, kernel, iterations=1)
        #
        # ## for debug
        # cv2.imshow('mask_sat', mask_sat)
        # cv2.waitKey(0)

        return mask_hue, mask_sat
        

    def __hue_mask(self, input_image, thresh_lower: int, thresh_upper: int):
        bgrLower = np.array([thresh_lower, 0, 0])    # maskする色の下限
        bgrUpper = np.array([thresh_upper, 255, 255])    # maskする色の上限
        mask = cv2.inRange(input_image, bgrLower, bgrUpper) # マスク作成
        return mask

    def __sat_mask(self, input_image, thresh_lower: int, thresh_upper: int):
        bgrLower = np.array([0, thresh_lower, 0])    # maskする色の下限
        bgrUpper = np.array([255, thresh_upper, 255])    # maskする色の上限
        mask = cv2.inRange(input_image, bgrLower, bgrUpper) # マスク作成
        return mask

if __name__ == "__main__":

    mask = Mask()

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

    last_path = tag + '/' + category_path

    if not os.path.exists(dataset_path + '/mask/' + last_path):
        os.makedirs(dataset_path + '/mask/' + last_path)

    base_dir=dataset_path + '/preprocessed/' + last_path
    save_dir=dataset_path + '/mask/' + last_path

    base_paths = os.listdir(base_dir)
    for filename in base_paths:
        if os.path.splitext(filename)[-1] not in ['.jpg']:
            continue
        input_image = cv2.imread(base_dir + '/' + filename)
        ## mainの実行
        mask_hue, mask_sat = mask.main(input_image)

        ## 元画像をmask
        masked_image = cv2.bitwise_and(input_image, input_image, mask=mask_hue)
        masked_image = cv2.bitwise_and(masked_image, masked_image, mask=mask_sat)

        ## for debug
        # cv2.imshow('masked', masked_image)
        # cv2.waitKey(0)

        cv2.imwrite(save_dir + '/' + filename, masked_image)

