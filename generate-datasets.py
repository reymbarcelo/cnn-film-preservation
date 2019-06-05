import os
import random
from shutil import copyfile

flic_dir = 'FLIC/full/'
damaged_dir = 'FLIC/damaged/'

train_dir = 'FLIC/train/'
val_dir = 'FLIC/val/'
test_dir = 'FLIC/test/'

NUM_TRAIN = 200
NUM_VAL = 50
NUM_TEST = 50

flic_filenames = os.listdir(flic_dir)
random.shuffle(flic_filenames)

# Make train data
for flic_filename in flic_filenames[:NUM_TRAIN]:
	scratch_num = random.randint(1, 5)
	new_filename = flic_filename[:-4] + '-scratch' + str(scratch_num) + '.png'
	# Copy non-damaged image
	copyfile(flic_dir + flic_filename, train_dir + 'original/' + new_filename)
	# Copy damaged image
	copyfile(damaged_dir + new_filename, train_dir + 'noisy/' + new_filename)
	print('Train:', flic_filename)

# Make val data
for flic_filename in flic_filenames[NUM_TRAIN:NUM_TRAIN+NUM_VAL]:
	scratch_num = random.randint(1, 5)
	new_filename = flic_filename[:-4] + '-scratch' + str(scratch_num) + '.png'
	# Copy non-damaged image
	copyfile(flic_dir + flic_filename, val_dir + 'original/' + new_filename)
	# Copy damaged image
	copyfile(damaged_dir + new_filename, val_dir + 'noisy/' + new_filename)
	print('Val:', flic_filename)

# Make test data
for flic_filename in flic_filenames[NUM_TRAIN+NUM_VAL:NUM_TRAIN+NUM_VAL+NUM_TEST]:
	scratch_num = random.randint(1, 5)
	new_filename = flic_filename[:-4] + '-scratch' + str(scratch_num) + '.png'
	# Copy non-damaged image
	copyfile(flic_dir + flic_filename, test_dir + 'original/' + new_filename)
	# Copy damaged image
	copyfile(damaged_dir + new_filename, test_dir + 'noisy/' + new_filename)
	print('Test:', flic_filename)
