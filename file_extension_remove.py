# -*- coding: utf-8 -*-
import argparse
import os
import imghdr
import re

parser=argparse.ArgumentParser()
parser.add_argument('--check_dir',type=str,default='no_dir',help='check directory path')
parser.add_argument('--extension',type=str,default='no_file_extension',help='check extension')
parser.add_argument('--without_mode',action='store_true')
parser.add_argument('--bom_mode',action='store_true')
parser.add_argument('--remove_mode',action='store_true')
args=parser.parse_args()

check_dir=args.check_dir
bom=b'\xef\xbb\xbf'

if not os.path.exists(check_dir):
	print("Image directory '" + check_dir + "' not found.")

if args.without_mode!=True:
	for (dir,subs,files) in os.walk(check_dir):
		for file in files:
			target=os.path.join(dir,file)
			if os.path.isfile(target):
				if f'{args.extension}'==os.path.splitext(os.path.basename(target))[1]:
					name=os.path.basename(target)
					print(target)
					if args.remove_mode==True:
						os.remove(target)

if args.without_mode==True:
	for (dir,subs,files) in os.walk(check_dir):
		for file in files:
			target=os.path.join(dir,file)
			if os.path.isfile(target):
				if f'{args.extension}'!=os.path.splitext(os.path.basename(target))[1]:
					name=os.path.basename(target)
					print(target)
					if args.remove_mode==True:
						os.remove(target)

if args.bom_mode==True:
	for (dir,subs,files) in os.walk(check_dir):
		for file in files:
			target=os.path.join(dir,file)
			if os.path.isfile(target):
				if imghdr.what(target)!=None:
					name=os.path.basename(target)
					if re.search(bom.decode('utf-8'),name)!=None:
						print(target)
						if args.remove_mode==True:
							os.remove(target)
				
