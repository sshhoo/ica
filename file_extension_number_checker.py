import os
import argparse
import re
import shutil

parser=argparse.ArgumentParser()
parser.add_argument('--check_dir',type=str,default='_',help='Path to folders of images.')
parser.add_argument('--extension',type=str,default='no_file_extension',help='check extension')
parser.add_argument('--subdir_mode',action='store_true')
parser.add_argument('--check_number',type=int,default=-1,help='check file number')
args=parser.parse_args()

extension=(args.extension).split(',')
dir_list=[]
cnum_list=[]
file_extension_dict={}
subdir_number_dict={}
check_number_dict={}

for i in range(len(extension)):
	file_extension_dict[extension[i]]=0

#拡張子付き
if extension[0]!='no_file_extension':
	for i in range(len(extension)):
		for (dir,subs,files) in os.walk(args.check_dir):
			for file in files:
				target=os.path.join(dir,file)
				if os.path.isfile(target):
					if re.search(f'{extension[i]}',os.path.basename(target))!=None:
						file_extension_dict[extension[i]]+=1
						if not dir in dir_list:
							dir_list.append(dir)
							for j in range(len(extension)):
								if j==0:
									subdir_number_dict[dir]={extension[j]:0}
								else:
									subdir_number_dict[dir].update({extension[j]:0})
						subdir_number_dict[dir][extension[i]]+=1
	if args.subdir_mode==True:
		for i in range(len(dir_list)):
			print(dir_list[i])
			cnum=0
			for j in range(len(extension)):
				print(extension[j])
				print(subdir_number_dict[dir_list[i]][extension[j]])
				print('')
				if args.check_number!=subdir_number_dict[dir_list[i]][extension[j]]:
					cnum_list.append(dir_list[i])
					if cnum==0:
						check_number_dict[dir_list[i]]={extension[j]:subdir_number_dict[dir_list[i]][extension[j]]}
						cnum+=1
					else:
						check_number_dict[dir_list[i]].update({extension[j]:subdir_number_dict[dir_list[i]][extension[j]]})
	if args.check_number!=-1:
		print('different dir\'s name')
		cnum_list=list(set(cnum_list))
		for i in range(len(cnum_list)):
			print(cnum_list[i])
			print(check_number_dict[cnum_list[i]])
			print('')
	print('check_dir')	
	print(args.check_dir)
	print('')
	for i in range(len(extension)):
		print(extension[i])
		print(file_extension_dict[extension[i]])
		print('')

#なし
else:
	for (dir,subs,files) in os.walk(args.check_dir):
		for file in files:
			target=os.path.join(dir,file)
			if os.path.isfile(target):
				file_extension_dict[extension[i]]+=1
				if not dir in dir_list:
					dir_list.append(dir)
					subdir_number_dict[dir]={extension[0]:0}
				subdir_number_dict[dir][extension[0]]+=1
	if args.subdir_mode==True:
		for i in range(len(dir_list)):
			print(dir_list[i])
			cnum=0
			print('file_number')
			print(subdir_number_dict[dir_list[i]][extension[0]])
			print('')
			if args.check_number!=subdir_number_dict[dir_list[i]][extension[0]]:
				cnum_list.append(dir_list[i])
				if cnum==0:
					check_number_dict[dir_list[i]]={extension[0]:subdir_number_dict[dir_list[i]][extension[0]]}
					cnum+=1
				else:
					check_number_dict[dir_list[i]].update({extension[0]:subdir_number_dict[dir_list[i]][extension[0]]})
	if args.check_number!=-1:
		print('different dir\'s name')
		cnum_list=list(set(cnum_list))
		for i in range(len(cnum_list)):
			print(cnum_list[i])
			print(check_number_dict[cnum_list[i]][extension[0]])
			print('')
	print('check_dir')	
	print(args.check_dir)
	print('')
	print('file_number')
	print(file_extension_dict[extension[0]])
	print('')

