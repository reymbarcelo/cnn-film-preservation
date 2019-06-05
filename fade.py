import os
import random
import cv2
import numpy as np

flic_dir = '../damaged_50/scratched/'
damaged_dir = '../damaged_50/noisy/'

flic_filenames = os.listdir(flic_dir)
scratch_filenames = os.listdir(damaged_dir)


for flic_filename in flic_filenames:
	m =  cv2.imread(flic_dir + flic_filename)

	# Get the size of the image
	h,w,bpp = np.shape(m)

	offsets = random.sample(range(16, 48), 3)

	# Process every pixel
	for x in range(h):
	   for y in range(w):
	       current_color = m[x][y]
	       
	       m[x][y][0] = max(0, m[x][y][0] - offsets[0])
	       m[x][y][1] = max(0, m[x][y][1] - offsets[1])
	       m[x][y][2] = max(0, m[x][y][2] - offsets[2])

	cv2.imwrite(damaged_dir + flic_filename,m)
	print('Successfully wrote', (damaged_dir + flic_filename))
