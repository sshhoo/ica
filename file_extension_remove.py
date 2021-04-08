import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument('--image_dir',type=str,default='no_dir',help='check directory path')
parser.add_argument('--extension',type=str,default='no_file_extension',help='check extension')
parser.add_argument('--without_mode',action='store_true')
parser.add_argument('--remove_mode',action='store_true')
args=parser.parse_args()

if args.image_dir!='_':
	if args.image_dir[-1]!='/':
		print('Add / as the last character, please.')
		print(args.image_dir)
		sys.exit(1)

if args.without_mode!=True:
	for (dir,subs,files) in os.walk(args.image_dir):
		for file in files:
			target=os.path.join(dir,file)
			if os.path.isfile(target):
				if f'{args.extension}'==os.path.splitext(os.path.basename(target))[1]:
					name=os.path.basename(target)
					print(target)
					if args.remove_mode==True:
						os.remove(target)

if args.without_mode==True:
	for (dir,subs,files) in os.walk(args.image_dir):
		for file in files:
			target=os.path.join(dir,file)
			if os.path.isfile(target):
				if f'{args.extension}'!=os.path.splitext(os.path.basename(target))[1]:
					name=os.path.basename(target)
					print(target)
					if args.remove_mode==True:
						os.remove(target)
