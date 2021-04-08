import os
import argparse
import re
import imghdr
import shutil

parser=argparse.ArgumentParser()
parser.add_argument('--image_dir',type=str,default='_',help='Path to folders of images.')
parser.add_argument('--tvt_rate',type=str,default='0.7,0.2,0.1',help='train,validation,test split rate')
args=parser.parse_args()

if args.image_dir!='_':
	if args.image_dir[-1]!='/':
		print('Add / as the last character, please.')
		print(args.image_dir)
		sys.exit(1)

tvt_list=args.tvt_rate.split(',')
print(tvt_list)
train_rate=float(tvt_list[0])
validation_rate=float(tvt_list[1])
test_rate=float(tvt_list[2])
print(train_rate)
print(validation_rate)
print(test_rate)

subdir_list=[f for f in os.listdir(args.image_dir) if os.path.isdir(os.path.join(args.image_dir,f))]

for i in range(len(subdir_list)):
	print(len(os.listdir(args.image_dir+subdir_list[i]+'/')))
	train_file_number=int(len(os.listdir(args.image_dir+subdir_list[i]+'/'))*train_rate)
	validation_file_number=int(len(os.listdir(args.image_dir+subdir_list[i]+'/'))*validation_rate)
	test_file_number=int(len(os.listdir(args.image_dir+subdir_list[i]+'/'))*test_rate)
	print(train_file_number)
	print(validation_file_number)
	print(test_file_number)
	c=0
	f=0
	train_dir_path=args.image_dir[:-1]+'_train/'+subdir_list[i]+'/'
	validation_dir_path=args.image_dir[:-1]+'_validation/'+subdir_list[i]+'/'
	test_dir_path=args.image_dir[:-1]+'_test/'+subdir_list[i]+'/'
	print(train_dir_path)
	print(validation_dir_path)
	print(test_dir_path)
	try:
		shutil.rmtree(train_dir_path)
		print('remove '+train_dir_path)
		os.makedirs(train_dir_path)
	except:
		os.makedirs(train_dir_path)
	try:
		shutil.rmtree(validation_dir_path)
		print('remove '+validation_dir_path)
		os.makedirs(validation_dir_path)
	except:
		os.makedirs(validation_dir_path)
	try:
		shutil.rmtree(test_dir_path)
		print('remove '+test_dir_path)
		os.makedirs(test_dir_path)
	except:
		os.makedirs(test_dir_path)
	for (dir,subs,files) in os.walk(args.image_dir+subdir_list[i]+'/'):
		for file in files:
			target=os.path.join(dir,file)
			if os.path.isfile(target):
				if imghdr.what(target)!=None:
					if f==0:
						if train_file_number==0:
							c=0
							f+=1
							continue
						shutil.copy(target,train_dir_path)
						c+=1
						if c==train_file_number:
							c=0
							f+=1
					if f==1:
						if validation_file_number==0:
							c=0
							f+=1
							continue
						shutil.copy(target,validation_dir_path)
						c+=1
						if c==validation_file_number:
							c=0
							f+=1
					if f==2:
						if test_file_number==0:
							c=0
							f+=1
							continue
						shutil.copy(target,test_dir_path)
						c+=1
						if c==test_file_number:
							c=0
							f+=1
					else:
						continue

