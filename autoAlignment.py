import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

def auto_alignment(size, data_paths, save_paths=None, top=None, left=None):
	if save_paths == None:
		save_paths = list(map((lambda path: f'result{os.sep}{path}'), data_paths))
		savedir_paths = set(map((lambda file: os.path.dirname(file)), save_paths))
		for savedir_path in savedir_paths:
			if not os.path.exists(savedir_path):
				os.makedirs(savedir_path)

	# 基準画像の読み込み
	base_image = cv2.imread(data_paths[0])
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
		print(f'Error: size {size} is bigger than the image size ({h},{w})')
		return

	# 基準画像の切り取り、保存
	base_image = base_image[top:top+size, left:left+size]  
	cv2.imwrite(save_paths[0], base_image)

	# 探索範囲を決定
	search_size = min(int(size*1.5), h, w)
	search_top = top+(size-search_size)//2
	if search_size == h or search_top<0:
		search_top = 0
	search_left = left+(size-search_size)//2
	if search_size == w or search_left<0:
		search_left = 0

	for data_path, save_path in zip(data_paths[1:], save_paths[1:]):
		# データ読み込み
		image = cv2.imread(data_path)
		trim_image = image[search_top:search_top+search_size, search_left:search_left+search_size]
		
		# 位置合わせ
		max_vals = []
		max_locs = []
		center = (search_size//2, search_size//2) # 回転中心
		angles = np.arange(-5., 5., 0.5).tolist()
		for angle in angles:
			trans = cv2.getRotationMatrix2D(center, angle , 1)
			rot_img = cv2.warpAffine(trim_image, trans, (search_size, search_size))
			res = cv2.matchTemplate(rot_img, base_image, cv2.TM_CCORR_NORMED)
			_, max_val, _, max_loc = cv2.minMaxLoc(res)
			max_vals.append(max_val)
			max_locs.append(max_loc)
		max_idx = np.argmax(np.array(max_vals))
		max_angle = angles[max_idx]
		max_loc = max_locs[max_idx]
		
		# 画像の回転と切り出し、保存
		trans = cv2.getRotationMatrix2D(center, max_angle , 1)
		result = cv2.warpAffine(trim_image, trans, (search_size, search_size))[max_loc[1]:max_loc[1]+size, max_loc[0]:max_loc[0]+size]
		cv2.imwrite(save_path, result)