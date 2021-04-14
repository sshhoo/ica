import os
import argparse
import imghdr
from PIL import Image
from PIL import ImageFile

import tensorflow as tf
assert tf.__version__.startswith('2')

parser=argparse.ArgumentParser()
parser.add_argument('--file_number',type=int,default=10000000000000,help='file number')
parser.add_argument('--image_dir',type=str,default='_',help='Path to folders of images.')
parser.add_argument('--resize_int',type=int,default=-1,help='resize integer')
args=parser.parse_args()

#Image loadable
ImageFile.LOAD_TRUNCATED_IMAGES=True

if args.image_dir!='_':
	if args.image_dir[-1]!='/':
		print('Add / as the last character, please.')
		print(args.image_dir)
		sys.exit(1)

if args.resize_int!=-1:
	resize_dir_name=args.image_dir.split('/')[0]
	if not os.path.exists(f'{args.resize_int}_resized_{resize_dir_name}/'):
		os.makedirs(f'{args.resize_int}_resized_{resize_dir_name}/')
	for (dir,subs,files) in os.walk(args.image_dir):
		if dir==args.image_dir:
			subdir_list=subs
	for i in range(len(subdir_list)):
		if not os.path.exists(f'{args.resize_int}_resized_{resize_dir_name}/{subdir_list[i]}/'):
			os.makedirs(f'{args.resize_int}_resized_{resize_dir_name}/{subdir_list[i]}/')
	for (dir,subs,files) in os.walk(args.image_dir):
			count=0
			for file in files:
				target=os.path.join(dir,file)
				if os.path.isfile(target):
					if imghdr.what(target)!=None:
						target_list=target.split('/')
						target_list[0]=f'{args.resize_int}_resized_{resize_dir_name}'
						if os.path.isfile('/'.join(target_list)):
							continue
						try:
							#tensorflow
							img_raw=tf.io.read_file(target)
							img_tensor=tf.image.decode_image(img_raw)
							if count>=args.file_number:
								print(target)
								os.remove(target)
							else:
								try:
									#Pillow
									img=Image.open(target)
									img_resize=img.resize((args.resize_int,args.resize_int))
									try:
										img_resize.save('/'.join(target_list))
										try:
											#resize tensorflow
											img_raw_resize=tf.io.read_file('/'.join(target_list))
											img_tensor_resize=tf.image.decode_image(img_raw_resize)
											count+=1
										except:
											print('/'.join(target_list))
											os.remove('/'.join(target_list))
									except:
										print(target)
										os.remove(target)
								except:
									print(target)
									os.remove(target)
						except:
							print(target)
							os.remove(target)
					else:
						print(target)
						os.remove(target)
