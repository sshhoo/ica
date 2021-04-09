import numpy as np
import os
from PIL import Image
import argparse
import imghdr
import sys
import datetime
import time
import copy

import tensorflow as tf
assert tf.__version__.startswith('2')

parser=argparse.ArgumentParser()
parser.add_argument('--image_dir',type=str,default='_',help='same directory making tfrecord')
parser.add_argument('--tfr_train_dir',type=str,default='_',help='Path to folders of tfrecord(train).')
parser.add_argument('--tfr_validation_dir',type=str,default='_',help='Path to folders of tfrecord(validation).')
parser.add_argument('--tfr_test_dir',type=str,default='_',help='Path to folders of tfrecord(test).')
parser.add_argument('--epochs',type=int,default=30,help='epoch size')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--mf',type=str,default='mobilenetv2',help='model flag')
parser.add_argument('--mw',type=str,default='imagenet',help='model weight')
parser.add_argument('--mtrain_mode',action='store_true',help='trainable model mode')
parser.add_argument('--op',type=str,default='sgd',help='optimizer')
parser.add_argument('--loss',type=str,default='categorical_crossentropy',help='loss function')
parser.add_argument('--original_mode',action='store_true',help='original model mode')
parser.add_argument('--tensorflow_mode',action='store_true',help='tensorflow model mode')
args=parser.parse_args()

AUTOTUNE=tf.data.experimental.AUTOTUNE

ctime='{0:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())
add_character=copy.deepcopy(ctime)+'\n'

if args.image_dir!='_':
	if args.image_dir[-1]!='/':
		print('Add / as the last character, please.')
		print(args.image_dir)
		sys.exit(1)

if args.tfr_train_dir!='_':
	if args.tfr_train_dir[-1]!='/':
		print('Add / as the last character, please.')
		print(args.tfr_train_dir)
		sys.exit(1)

if args.tfr_validation_dir!='_':
	if args.tfr_validation_dir[-1]!='/':
		print('Add / as the last character, please.')
		print(args.tfr_validation_dir)
		sys.exit(1)

if args.tfr_test_dir!='_':
	if args.tfr_test_dir[-1]!='/':
		print('Add / as the last character, please.')
		print(args.tfr_test_dir)
		sys.exit(1)

if not os.path.exists('model_result/'):
	os.makedirs('model_result/')

if not os.path.exists("speed/"):
	os.makedirs("speed/")

if not os.path.exists(f'model_result/{os.path.dirname(args.tfr_test_dir).split("/")[-1]}_evaluate.txt'):
	with open(f'model_result/{os.path.dirname(args.tfr_test_dir).split("/")[-1]}_evaluate.txt',"w"):pass

if not os.path.exists("speed/model_speed.txt"):
	with open("speed/model_speed.txt","w"):pass

if args.mw=='None':
	args.mw=None

train_list=[]

for (dir,subs,files) in os.walk(args.image_dir):
	for file in files:
		target=os.path.join(dir,file)
		if os.path.isfile(target):
			if imghdr.what(target)!=None:
				if len(train_list)==0:
					#pillow check
					img=Image.open(target)
					#tensorflow check
					img_raw=tf.io.read_file(target)
					img_tensor=tf.image.decode_image(img_raw)
					height,width,depth=img_tensor.shape[0],img_tensor.shape[1],img_tensor.shape[2]
				train_list.append(target)

train_list.sort()
train_label=[]
for i in range(len(train_list)):
	if not os.path.basename(os.path.dirname(train_list[i])) in train_label:
		train_label.append(os.path.basename(os.path.dirname(train_list[i])))

print('train list length')
print(len(train_list))
print('train label length')
print('class number')
print(len(train_label))

label_to_index=dict((name,index) for index,name in enumerate(train_label))
index_to_label=dict((index,name) for index,name in enumerate(train_label))

print('height')
print(height)
print('width')
print(width)
print('depth')
print(depth)

#tfrecord of train
tfr_train_list=[]
for (dir,subs,files) in os.walk(args.tfr_train_dir):
	for file in files:
		target=os.path.join(dir,file)
		if os.path.isfile(target):
			tfr_train_list.append(target)

if args.tfr_validation_dir!='_':
	#tfrecord of validation
	tfr_validation_list=[]
	for (dir,subs,files) in os.walk(args.tfr_validation_dir):
		for file in files:
			target=os.path.join(dir,file)
			if os.path.isfile(target):
				tfr_validation_list.append(target)

#tfrecord of test
tfr_test_list=[]
for (dir,subs,files) in os.walk(args.tfr_test_dir):
	for file in files:
		target=os.path.join(dir,file)
		if os.path.isfile(target):
			tfr_test_list.append(target)

#learning preparation
def parse_example(example):
	feature=tf.io.parse_example(
		[example],
		features={
			'image_raw':tf.io.FixedLenFeature([],dtype=tf.string),
			'label':tf.io.FixedLenFeature([],dtype=tf.string),
		}
	)
	
	image=tf.reshape(tf.io.decode_raw(feature['image_raw'],tf.float32),tf.stack([height,width,depth]))
	#one-hot→dimension
	label=tf.reshape(tf.io.decode_raw(feature['label'],tf.float32),tf.stack([len(train_label)]))
	
	return image,label

steps_per_epoch=tf.math.ceil(len(train_list)/args.batch_size).numpy()
print(steps_per_epoch)

dataset_train=tf.data.Dataset.from_tensor_slices([f'tfrecord/{os.path.dirname(args.tfr_train_dir).split("/")[-1]}/{os.path.dirname(args.tfr_train_dir).split("/")[-1]}.tfrecord.{str(i)}' for i in range(len(tfr_train_list))]).interleave(lambda filename: tf.data.TFRecordDataset(filename,compression_type='GZIP').map(parse_example,num_parallel_calls=AUTOTUNE),cycle_length=len(tfr_train_list)).shuffle(buffer_size=int(steps_per_epoch)).batch(args.batch_size).prefetch(buffer_size=AUTOTUNE).repeat(-1)
dataset_test=tf.data.Dataset.from_tensor_slices([f'tfrecord/{os.path.dirname(args.tfr_test_dir).split("/")[-1]}/{os.path.dirname(args.tfr_test_dir).split("/")[-1]}.tfrecord.{str(i)}' for i in range(len(tfr_test_list))]).interleave(lambda filename: tf.data.TFRecordDataset(filename,compression_type='GZIP').map(parse_example,num_parallel_calls=AUTOTUNE),cycle_length=len(tfr_test_list)).batch(args.batch_size).prefetch(buffer_size=AUTOTUNE)

if args.tfr_validation_dir!='_':
	dataset_validation=tf.data.Dataset.from_tensor_slices([f'tfrecord/{os.path.dirname(args.tfr_validation_dir).split("/")[-1]}/{os.path.dirname(args.tfr_validation_dir).split("/")[-1]}.tfrecord.{str(i)}' for i in range(len(tfr_validation_list))]).interleave(lambda filename: tf.data.TFRecordDataset(filename,compression_type='GZIP').map(parse_example,num_parallel_calls=AUTOTUNE),cycle_length=len(tfr_validation_list)).shuffle(buffer_size=int(steps_per_epoch)).batch(args.batch_size).prefetch(buffer_size=AUTOTUNE)

def trained_model(mf,mw,height,width,depth,mtrain_mode,train_label):
	if mf=='densenet121':
		f_model=tf.keras.applications.DenseNet121(include_top=False,weights=mw,input_shape=(height,width,depth))
	if mf=='densenet169':
		f_model=tf.keras.applications.DenseNet169(include_top=False,weights=mw,input_shape=(height,width,depth))
	if mf=='densenet201':
		f_model=tf.keras.applications.DenseNet201(include_top=False,weights=mw,input_shape=(height,width,depth))
	#below tensorflow 2.3 doesn't work
	if mf=='efficientnetb0':
		f_model=tf.keras.applications.EfficientNetB0(include_top=False,weights=mw,input_shape=(height,width,depth))
	#below tensorflow 2.3 doesn't work
	if mf=='efficientnetb1':
		f_model=tf.keras.applications.EfficientNetB1(include_top=False,weights=mw,input_shape=(height,width,depth))
	#below tensorflow 2.3 doesn't work
	if mf=='efficientnetb2':
		f_model=tf.keras.applications.EfficientNetB2(include_top=False,weights=mw,input_shape=(height,width,depth))
	#below tensorflow 2.3 doesn't work
	if mf=='efficientnetb3':
		f_model=tf.keras.applications.EfficientNetB3(include_top=False,weights=mw,input_shape=(height,width,depth))
	#below tensorflow 2.3 doesn't work
	if mf=='efficientnetb4':
		f_model=tf.keras.applications.EfficientNetB4(include_top=False,weights=mw,input_shape=(height,width,depth))
	#below tensorflow 2.3 doesn't work
	if mf=='efficientnetb5':
		f_model=tf.keras.applications.EfficientNetB5(include_top=False,weights=mw,input_shape=(height,width,depth))
	#below tensorflow 2.3 doesn't work
	if mf=='efficientnetb6':
		f_model=tf.keras.applications.EfficientNetB6(include_top=False,weights=mw,input_shape=(height,width,depth))
	#below tensorflow 2.3 doesn't work
	if mf=='efficientnetb7':
		f_model=tf.keras.applications.EfficientNetB7(include_top=False,weights=mw,input_shape=(height,width,depth))
	if mf=='inceptionresnetv2':
		f_model=tf.keras.applications.InceptionResNetV2(include_top=False,weights=mw,input_shape=(height,width,depth))
	if mf=='inceptionv3':
		f_model=tf.keras.applications.InceptionV3(include_top=False,weights=mw,input_shape=(height,width,depth))
	if mf=='mobilenet':
		f_model=tf.keras.applications.MobileNet(include_top=False,weights=mw,input_shape=(height,width,depth))
	if mf=='mobilenetv2':
		f_model=tf.keras.applications.MobileNetV2(include_top=False,weights=mw,input_shape=(height,width,depth))
	if mf=='mobilenetv3large':
		f_model=tf.keras.applications.MobileNetV3Large(include_top=False,weights=mw,input_shape=(height,width,depth))
	if mf=='mobilenetv3small':
		f_model=tf.keras.applications.MobileNetV3Small(include_top=False,weights=mw,input_shape=(height,width,depth))
	if mf=='nasnetlarge':
		f_model=tf.keras.applications.NASNetLarge(include_top=False,weights=mw,input_shape=(height,width,depth))
	if mf=='nasnetmobile':
		f_model=tf.keras.applications.NASNetMobile(include_top=False,weights=mw,input_shape=(height,width,depth))
	if mf=='resnet101':
		f_model=tf.keras.applications.ResNet101(include_top=False,weights=mw,input_shape=(height,width,depth))
	if mf=='resnet101v2':
		f_model=tf.keras.applications.ResNet101V2(include_top=False,weights=mw,input_shape=(height,width,depth))
	if mf=='resnet152':
		f_model=tf.keras.applications.ResNet152(include_top=False,weights=mw,input_shape=(height,width,depth))
	if mf=='resnet152v2':
		f_model=tf.keras.applications.ResNet152V2(include_top=False,weights=mw,input_shape=(height,width,depth))
	if mf=='resnet50':
		f_model=tf.keras.applications.ResNet50(include_top=False,weights=mw,input_shape=(height,width,depth))
	if mf=='resnet50v2':
		f_model=tf.keras.applications.ResNet50V2(include_top=False,weights=mw,input_shape=(height,width,depth))
	if mf=='vgg16':
		f_model=tf.keras.applications.VGG16(include_top=False,weights=mw,input_shape=(height,width,depth))
	if mf=='vgg19':
		f_model=tf.keras.applications.VGG19(include_top=False,weights=mw,input_shape=(height,width,depth))
	if mf=='xception':
		f_model=tf.keras.applications.Xception(include_top=False,weights=mw,input_shape=(height,width,depth))
	print('current model')
	print(mf)
	f_model.summary()
	if mtrain_mode==False:
		for layer in f_model.layers[:-1]:
			layer.trainable=False
	model=tf.keras.Sequential([
		f_model,
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(1024,activation='relu'),
		tf.keras.layers.Dense(len(train_label),activation='softmax')
	])
	
	return model

def original_model(height,width,depth,train_label):
	#LeNet5
	model=tf.keras.Sequential([
		tf.keras.layers.Conv2D(6,(5,5),activation='relu',padding='same',input_shape=(height,width,depth)),
		tf.keras.layers.MaxPooling2D((2,2)),
		tf.keras.layers.Conv2D(16,(5,5),activation='relu',padding='valid'),
		tf.keras.layers.MaxPooling2D((2,2)),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(120,activation='relu'),
		tf.keras.layers.Dense(84,activation='relu'),
		tf.keras.layers.Dense(len(train_label),activation='softmax')
	])
	
	return model

if args.tensorflow_mode!=True:
	if args.original_mode!=True:
		model=trained_model(args.mf,args.mw,height,width,depth,args.mtrain_mode,train_label)
	else:
		model=original_model(height,width,depth,train_label)
else:
	pass

model.compile(optimizer=args.op,loss=args.loss,metrics=['accuracy'])
model.summary()

train_log_path=f'{os.path.dirname(args.tfr_train_dir).split("/")[-1]}'
test_log_path=f'{os.path.dirname(args.tfr_test_dir).split("/")[-1]}'

callbacks=[
	tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',patience=2,verbose=1),
	tf.keras.callbacks.EarlyStopping(monitor='loss',patience=5,verbose=1),
	tf.keras.callbacks.TensorBoard(log_dir='log_dir/'+train_log_path+'-'+test_log_path+'-'+str(args.batch_size)+'-'+str(args.op)+'-'+str(args.mf)+'-'+str(args.mw)+'/',histogram_freq=1)
]

bt=time.time()

if args.tfr_validation_dir!='_':
	history=model.fit(
		dataset_train,
		validation_data=dataset_validation,
		epochs=args.epochs,
		steps_per_epoch=steps_per_epoch,
		callbacks=callbacks
	)

else:
	history=model.fit(
		dataset_train,
		validation_data=dataset_test,
		epochs=args.epochs,
		steps_per_epoch=steps_per_epoch,
		callbacks=callbacks
	)

rt=time.time()-bt
add_character=add_character+f"{os.path.dirname(args.image_dir).split('/')[-1]}-"+str(args.batch_size)+'-'+str(args.op)+'-'+str(args.mf)+': '+str(rt)+' sec'+'\n\n'

print('テストデータ評価')
model.evaluate(dataset_test,verbose=1)

with open(f'model_result/{os.path.dirname(args.tfr_test_dir).split("/")[-1]}_evaluate.txt',mode='a') as f:
	f.write(f'{ctime}\n')
	f.write(f'{os.path.dirname(args.tfr_train_dir).split("/")[-1]}\n')
	f.write('loss,accuracy\n')
	f.write(f'{str(model.evaluate(dataset_test,verbose=1))}\n')
	f.write('\n')

with open("speed/model_speed.txt","a") as f:
	f.write(add_character)

predictions=model.predict(dataset_test,verbose=1)

c=0
for i in dataset_test:
	if c==0:
		initial_data=i[1].numpy()
		c=1

for i in range(len(predictions)):
	if i<args.batch_size:
		print(i)
		print(predictions[i])
		print(initial_data[i])
	else:
		break

converter=tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations=[tf.lite.Optimize.DEFAULT]
tflite_model=converter.convert()
open(f'./model_result/{os.path.basename(os.path.dirname(args.image_dir))}_model.tflite','wb').write(tflite_model)
with open(f'./model_result/{os.path.basename(os.path.dirname(args.image_dir))}_labels.txt',mode='w') as f:
	f.write('\n'.join(list(os.walk(args.image_dir))[0][1]))
