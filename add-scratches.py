from PIL import Image
import os
import random

flic_dir = 'FLIC/full/'
scratches_dir = 'FLIC/scratches/'
damaged_dir = 'FLIC/damaged/'

flic_filenames = os.listdir(flic_dir)
scratch_filenames = os.listdir(scratches_dir)


for flic_filename in flic_filenames:
	for scratch_filename in scratch_filenames:
		background = Image.open(flic_dir + flic_filename)
		foreground = Image.open(scratches_dir + scratch_filename)

		background.paste(foreground, (0, 0), foreground)
		damaged_filename = damaged_dir + flic_filename[:-4] + '-' + scratch_filename
		background.save(damaged_filename)
		print('Saved file ', damaged_filename)
