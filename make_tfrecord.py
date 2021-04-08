import os
from PIL import Image
import argparse
import imghdr
import sys
import pathlib
import random

import tensorflow as tf
assert tf.__version__.startswith('2')

parser=argparse.ArgumentParser()
parser.add_argument('--image_dir',type=str,default='_',help='Path to folders of labeled images.')
parser.add_argument('--stn',type=int,default=10000,help='tfrecords per image number')
args=parser.parse_args()

if args.image_dir!='_':
	if args.image_dir[-1]!='/':
		print('Add / as the last character, please.')
		print(args.image_dir)
		sys.exit(1)

if not os.path.exists(f'tfrecord/{os.path.dirname(args.image_dir).split("/")[-1]}/'):
	os.makedirs(f'tfrecord/{os.path.dirname(args.image_dir).split("/")[-1]}/')

def image_example(image_path,label):
	img_raw=tf.io.read_file(image_path)
	img_tensor=tf.image.decode_image(img_raw)
		
	feature={
		'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[(img_tensor.numpy()/255).astype('float32').tostring()])),
		'label':tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.astype('float32').tostring()])),
	}
	
	return tf.train.Example(features=tf.train.Features(feature=feature))

train_list=[]

for (dir,subs,files) in os.walk(args.image_dir):
	for file in files:
		target=os.path.join(dir,file)
		if os.path.isfile(target):
			if imghdr.what(target)!=None:
				try:
					#pillow check
					img=Image.open(target)
					#tensorflow check
					img_raw=tf.io.read_file(target)
					img_tensor=tf.image.decode_image(img_raw)
					train_list.append(target)
				except:
					print('no readable')
					print(target)

train_list.sort()
train_label=[]
for i in range(len(train_list)):
	if not os.path.basename(os.path.dirname(train_list[i])) in train_label:
		train_label.append(os.path.basename(os.path.dirname(train_list[i])))

print('train list length')
print(len(train_list))
print('train label length')
print(len(train_label))

random.shuffle(train_list)
label_to_index=dict((name,index) for index,name in enumerate(train_label))
print(label_to_index)
all_image_labels=[label_to_index[pathlib.Path(path).parent.name] for path in train_list]
all_image_labels=tf.keras.utils.to_categorical(all_image_labels)
print(type(all_image_labels))
print(all_image_labels.shape)

#split tfrecords number
print(len(train_list)//args.stn)
stn,stn_mod=divmod(len(train_list),args.stn)
print(stn)
print(stn_mod)

for h in range(stn+1):
	with tf.io.TFRecordWriter(f'tfrecord/{os.path.dirname(args.image_dir).split("/")[-1]}/{os.path.dirname(args.image_dir).split("/")[-1]}.tfrecord.{str(h)}',tf.io.TFRecordOptions(compression_type='GZIP',compression_level=9)) as writer:
		print(f'start writing tfrecord/{os.path.dirname(args.image_dir).split("/")[-1]}/{os.path.dirname(args.image_dir).split("/")[-1]}.tfrecord.{str(h)}')
		if h!=stn:
			for i in range(args.stn):
				tf_example=image_example(train_list[i+h*args.stn],all_image_labels[i+h*args.stn])
				writer.write(tf_example.SerializeToString())
				print(f'done {i+h*args.stn+1}/{len(train_list)}')
		else:
			for i in range(stn_mod):
				tf_example=image_example(train_list[i+h*args.stn],all_image_labels[i+h*args.stn])
				writer.write(tf_example.SerializeToString())
				print(f'done {i+h*args.stn+1}/{len(train_list)}')
