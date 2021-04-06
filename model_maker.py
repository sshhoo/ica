import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import configs
from tflite_model_maker import image_classifier
from tflite_model_maker import ImageClassifierDataLoader
from tflite_model_maker import model_spec

import matplotlib.pyplot as plt
import numpy as np

import argparse
import os
import shutil
import pathlib
import random
import datetime

parser=argparse.ArgumentParser()
parser.add_argument('--input_dir',type=str,default='no',help='input directory, needs /')
parser.add_argument('--input_size',type=int,default=192,help='input size')
parser.add_argument('--epochs',type=int,default=30,help='epoch size')
parser.add_argument('--tv_rate',type=str,default='0.7,0.2',help='remaining test data ')
parser.add_argument('--pre_num',type=int,default=30,help='prediction_number')
parser.add_argument('--ft_mode',action='store_true')
parser.add_argument('--f_model_mode',action='store_true')
parser.add_argument('--channum',type=int,default=3,help='channel number')
args=parser.parse_args()

INPUT_SIZE=args.input_size
AUTOTUNE=tf.data.experimental.AUTOTUNE
BATCH_SIZE=32
time='{0:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())

def preprocess_image(image):
	image=tf.image.decode_jpeg(image,channels=args.channum)
	image=tf.image.resize(image,[INPUT_SIZE,INPUT_SIZE])
	image/=255.0	#normalize to [0,1] range
	
	return image

def load_and_preprocess_image(path):
	image=tf.io.read_file(path)
	
	return preprocess_image(image)

def load_and_preprocess_from_path_label(path,label):
	return load_and_preprocess_image(path),label
	
def generate_prefetch(image_path,label_path):
	image_ds=tf.data.Dataset.from_tensor_slices(image_path).map(load_and_preprocess_image,num_parallel_calls=AUTOTUNE)
	label_ds=tf.data.Dataset.from_tensor_slices(tf.cast(label_path,tf.int64))
	image_label_ds=tf.data.Dataset.from_tensor_slices((image_path,label_path)).map(load_and_preprocess_from_path_label)
	ds=image_label_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
	
	return ds

if not os.path.exists('model_result/'):
	os.makedirs('model_result/')

with open('./labels.txt',mode='w') as f:
	f.write('\n'.join(list(os.walk(args.input_dir))[0][1]))

idir_name=os.path.basename(os.path.dirname(args.input_dir))
print(idir_name)

data = ImageClassifierDataLoader.from_folder(args.input_dir)

plt.figure(figsize=(10,10))
for i, (image, label) in enumerate(data.dataset.take(25)):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(image.numpy(), cmap=plt.cm.gray)
	plt.xlabel(data.index_to_label[label.numpy()])
plt.savefig(f'model_result/{idir_name}.png')

if args.ft_mode==False:
	label=list(os.walk(args.input_dir))[0][1]
	print(label)
	for item in pathlib.Path(args.input_dir).iterdir():
		print(item)
	all_image_paths=list(pathlib.Path(args.input_dir).glob('*/*'))
	all_image_paths=[str(path) for path in all_image_paths]
	print(len(all_image_paths))
	random.shuffle(all_image_paths)
	label_to_index=dict((name,index) for index,name in enumerate(label))
	print(label_to_index)
	all_image_labels=[label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
	all_image_labels=tf.keras.utils.to_categorical(all_image_labels)
	print('First 10 labels indices: ',all_image_labels[:10])
	print('First 10 images indices: ',all_image_paths[:10])
	
	train_images=all_image_paths[:int(len(all_image_paths)*float(args.tv_rate.split(',')[0]))]
	train_labels=all_image_labels[:int(len(all_image_paths)*float(args.tv_rate.split(',')[0]))]
	validation_images=all_image_paths[int(len(all_image_paths)*float(args.tv_rate.split(',')[0])):int(len(all_image_paths)*(float(args.tv_rate.split(',')[0])+float(args.tv_rate.split(',')[1])))]
	validation_labels=all_image_labels[int(len(all_image_paths)*float(args.tv_rate.split(',')[0])):int(len(all_image_paths)*(float(args.tv_rate.split(',')[0])+float(args.tv_rate.split(',')[1])))]
	test_images=all_image_paths[int(len(all_image_paths)*(float(args.tv_rate.split(',')[0])+float(args.tv_rate.split(',')[1]))):]
	test_labels=all_image_labels[int(len(all_image_paths)*(float(args.tv_rate.split(',')[0])+float(args.tv_rate.split(',')[1]))):]
	
	train_ds=generate_prefetch(train_images,train_labels)
	validation_ds=generate_prefetch(validation_images,validation_labels)
	test_ds=generate_prefetch(test_images,test_labels)
	
	steps_per_epoch=tf.math.ceil(len(train_images)/BATCH_SIZE).numpy()
	print(steps_per_epoch)
	
	if args.f_model_mode==True:
		f_model=tf.keras.applications.DenseNet201(include_top=False,input_shape=(INPUT_SIZE,INPUT_SIZE,args.channum))
		f_model.summary()
		for layer in f_model.layers[:-1]:
			layer.trainable=False
	
		model=tf.keras.Sequential([
			f_model,
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(120,activation='relu'),
			tf.keras.layers.Dense(84,activation='relu'),
			tf.keras.layers.Dense(len(label),activation='softmax')
		])
	
	else:
		model=tf.keras.Sequential([
			tf.keras.layers.Conv2D(6,(5,5),activation='relu',padding='same',input_shape=(INPUT_SIZE,INPUT_SIZE,args.channum)),
			tf.keras.layers.MaxPooling2D((2,2)),
			tf.keras.layers.Conv2D(16,(5,5),activation='relu',padding='valid'),
			tf.keras.layers.MaxPooling2D((2,2)),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(120,activation='relu'),
			tf.keras.layers.Dense(84,activation='relu'),
			tf.keras.layers.Dense(len(label),activation='softmax')
		])
	model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
	model.summary()
	tf.keras.utils.plot_model(model,to_file=f'model_result/{idir_name}_model.png',show_shapes=True)
	
	callbacks=[
		tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',patience=2,verbose=1),
		tf.keras.callbacks.EarlyStopping(monitor='loss',patience=5,verbose=1),
		tf.keras.callbacks.TensorBoard(log_dir='log_dir',histogram_freq=1)
	]
	
	model.fit(
		train_ds,
		validation_data=validation_ds,
		epochs=args.epochs,
		steps_per_epoch=steps_per_epoch,
		callbacks=callbacks
	)
	
	print('テストデータ評価')
	model.evaluate(test_ds,verbose=1)
	
	with open(f'./model_result/{idir_name}_test_evaluate.txt',mode='a') as f:
		f.write(f'{time}\n')
		f.write('loss,accuracy\n')
		f.write(f'{str(model.evaluate(test_ds,verbose=1))}\n')
		f.write('\n')
	
	predictions=model.predict(test_ds)
	
	for i in range(len(predictions)):
		if i<=args.pre_num:
			print(i)
			print(predictions[i])
			print(all_image_labels[int(len(all_image_paths)*(float(args.tv_rate.split(',')[0])+float(args.tv_rate.split(',')[1])))+i])
	
	converter=tf.lite.TFLiteConverter.from_keras_model(model)
	converter.optimizations=[tf.lite.Optimize.DEFAULT]
	tflite_model=converter.convert()
	open(f'./model_result/{idir_name}_model.tflite','wb').write(tflite_model)
	shutil.move('./labels.txt',f'./model_result/{idir_name}_labels.txt')

if args.ft_mode==True:
	train_data,test_data = data.split(0.9)
	print(train_data)
	model = image_classifier.create(train_data)
	print('evaluate')
	loss, accuracy = model.evaluate(test_data)
	model.export(export_dir='model_result/',tflite_filename=f'{idir_name}_ft_model.tflite')
	shutil.move('./labels.txt',f'./model_result/{idir_name}_labels.txt')



