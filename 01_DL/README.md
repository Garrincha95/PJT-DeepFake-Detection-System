<h1>DeepFake Detection</h1>

---

<h3>목차</h3>

**[1. 프로젝트 소개](#1-프로젝트-소개)**

**[2. 코드 리뷰](#2-코드-리뷰)**

**[3. 시연 영상](#3-시연-영상)**

**[4. 배운 것](#4-배운-것)**

---

<h3>1. 프로젝트 소개</h3>

> **프로젝트 기간**: 2020.03.14. - 2020.04.30

> **스택**: Python

> **툴**: Jupyter Notebook

---

<h3>2. 코드 리뷰</h3>

1)  **Data classification.ipynb**

```python
# shutil : move 함수를 이용해, 새로 생성한 REAL과 FAKE 폴더로 영상을 이동시킨다
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

# Kaggle Deepfake Detection Challenge Dataset의 https://www.kaggle.com/c/deepfake-detection-challenge/data
# train_sample_videos 400개 영상과, 03.zip의 약 1500개 영상의 REAL, FAKE 정보가 담긴, json 파일로 구성
# 각각 metadata를 Dataframe으로 읽어온 뒤, 하나의 Dataframe으로 구성
# json 파일을 이용하여 barplot을 통해 영상 REAL, FAKE 비율 시각적으로 구성

metadata_0 = pd.read_json('./metadata.json').T
metadata_1 = pd.read_json('./metadata3.json').T

total_metadata = pd.concat([metadata_0, metadata_1])

metadata.groupby('label')['label'].count().plot(figsize=(12,5), kind='bar')
plt.show()

# 이진 분류를 위해, REAL 영상과 FAKE 영상 분리
# total_metadata의 label 값을 토대로 REAL인 영상 리스트(real)와 FAKE인 영상 리스트(fake) 생성
# REAL, FAKE 영상 리스트 앞에 영상이 있는 파일 경로를 붙여, 영상 경로 리스트 생성 (real_vid_path, fake_vid_path)
# shutil.move를 이용하여 REAL, FAKE 영상을 각 REAL, FAKE 폴더로 이동 (이동 전 REAL, FAKE 폴더 생성 요구)

real = total_metadata[total_metadata["label"] == "REAL"]
fake = total_metadata[total_metadata["label"] == "FAKE"]

real_vid_path = list(map(lambda x: 'C:/ai/workspace/sh/dfdc_train_part_3/'+x, json_real.index))
real_vid_path

fake_vid_path = list(map(lambda x: 'C:/ai/workspace/sh/dfdc_train_part_3/'+x, json_fake.index))
fake_vid_path

for i in real_vid_path:
    shutil.move(i, 'C:/ai/workspace/sh/REAL')
    
for i in fake_vid_path:
    shutil.move(i, 'C:/ai/workspace/sh/FAKE')    
```

---



2) **Face Crop by facenet_pytorch`s MTCNN**

```python
# MTCNN : 얼굴 검출 후 얼굴 크롭하는 라이브러리
import os
import cv2 as cv
import numpy as np
from facenet_pytorch import MTCNN
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]= "1"

# REAL과 FAKE 폴더에 속한 모든 폴더 경로 리스트 생성
real_dir = ('C:/ai/workspace/sh/REAL/')
real_list = [real_dir + x for x in os.listdir(real_dir)]
real_list = list(map(lambda x: x+"/", real_dir))

fake_dir = ('C:/ai/workspace/sh/FAKE/')
fake_list = [fake_dir + x for x in os.listdir(fake_dir)]
fake_list = list(map(lambda x: x+"/", fake_list))

# Face Detect
# v_cap : 영상을 프레임별로 자름
# v_len : 프레임의 총 개수를 변수에 저장
# cv2.VideoCapture.grab() : 프레임을 가져온다. 실패 시 success 변수엔 False가 저장
# cv2.VideoCapture.retrieve() : grab한 프레임을 decode한다. 프레임이 없을 경우 False가 success 변수에 저장
# cv.COLOR_BGR2RGB : 프레임을 BGR에서 RGB로 변환
# MTCNN : 얼굴을 검출하고, margin을 줘 얼굴이 살짝 잘리는 것을 방지
# save_path : 캡쳐 파일 저장
# count : 영상에서 프레임별로 검출되는 모든 얼굴에 대해 번호를 매기기 위함

    
    count=0
    file =  videos.split("/")[-1]
    fileName = file.split(".")[0]
    mtcnn = MTCNN(margin=50, keep_all=True, post_process=True, device='cuda:0',thresholds=[.9,.9,.9])
    v_cap = cv.VideoCapture(videos)
    v_len = int(v_cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    for i in tqdm(range(v_len)):    
        success = v_cap.grab()
        if i % 1 == 0:
            success, frame = v_cap.retrieve()
        else:
            continue
        if not success:
            continue
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        face = mtcnn(frame, save_path = f'C:/ai/workspace/sh/newtrain/{label}/{num}/{fileName}{count}.jpg')
        count=count+1

    v_cap.release()
    
# 사람 폴더별로 폴더 안 모든 영상에 대해 chimac 함수를 실행시킨다

for path in range(len(real_list)):
    real_vid = [real_list[path] + x for x in os.listdir(real_list[path])]
    for vid in real_vid:
        chimac(vid, path, 'REAL')
        

for path in range(len(fake_list)):
    fake_vid = [fake_list[path] + x for x in os.listdir(fake_list[path])]
    for vid in fake_vid:
        chimac(vid, path, 'FAKE')        
```

---



3) **Classification Balancing (Undersampling)**

```python
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import os
import pickle
import shutil
import matplotlib.pyplot as plt

# Real photo counting by person
# real_dir : 사람별 폴더 경로
# real_dir_list : 사람별 폴더 리스트
# real_count : 사람별 폴더 내 사진 개수

real_dir = ('C:/ai/workspace/sh/newtrain/REAL/')
real_dir_list = [real_dir + x for x in os.listdir(real_dir)]
real_dir_list = list(map(lambda x: x+"/", real_dir_list))

real_count=[]
for lists in real_dir_list:
    real_list = [lists + x for x in os.listdir(lists)]
    real_count.append(len(real_list))
    
# Fake photo counting by person
# fake_dir : 사람별 폴더 경로
# fake_dir_list : 사람별 폴더 리스트
# fake_count : 사람별 폴더 내 사진 개수

fake_dir = ('C:/ai/workspace/sh/newtrain/FAKE/') 
fake_dir_list = [fake_dir + x for x in os.listdir(fake_dir)]
fake_dir_list = list(map(lambda x: x+"/", fake_dir_list))

fake_count=[]
for lists in fake_dir_list:
    fake_list = [lists + x for x in os.listdir(lists)]
    fake_count.append(len(fake_list))
    
# Classification Balancing
# balance_fake : 사람별 폴더 경로
# balance_fake_list : 사람별 폴더 리스트
# shutil.copy() : 사람별로 real 개수만큼 fake에서 랜덤 추출 후 파일 복사

balance_fake = ('C:/ai/workspace/sh/newtrain/fakesamepic/') 
balance_fake_list = [balance_fake + x for x in os.listdir(balance_fake)]
balance_fake_list

np.random.seed(33)
for path in range(len(real_dir_list)):
    real = [real_dir_list[path] + x for x in os.listdir(real_dir_list[path])]
    fake = [fake_dir_list[path] + x for x in os.listdir(fake_dir_list[path])]
    bal_fake = np.random.choice(fake, len(real), replace=False)
    for bal in bal_fake:
        shutil.copy(bal, fakesamplefoldlist[path])
        
# Train, Validation, Test division by person
# 마지막 두번째 폴더의 사람은 test로 분류
# 나머지 사람은, 사람별로 약 9:1의 비율로 train과 validation 데이터로 분류
# 사람별 폴더에 들어있는 사진들의 리스트 순서를 임의로 섞은 후, 리스트 앞에서부터 사진 갯수의 0.11 비율에 해당하는 개수만큼을 validation으로, 나머지는 train으로 분류

for lists in real_dir_list[:-2]:
    real_list = [lists + x for x in os.listdir(lists)]
    real_list = np.random.choice(real_list, len(real_list), replace=False)
    for real in real_list[:int(len(real_list)*0.11)]:
        shutil.copy(real, 'C:/ai/workspace/sh/newtrain/val/real')
    for real in real_list[int(len(real_list)*0.11):]:
        shutil.copy(j, 'C:/ai/workspace/sh/newtrain/train/real')
        
for lists in fakesamefoldlist[:-2]:
    balance_fake_list = [lists + x for x in os.listdir(lists)]
    balance_fake_list = np.random.choice(balance_fake_list, len(balance_fake_list), replace=False)
    for fake in balance_fake_list[:int(len(balance_fake_list)*0.11)]:
        shutil.copy(fake, 'C:/ai/workspace/sh/newtrain/val/fake')
    for fake in balance_fake_list[int(len(balance_fake_list)*0.11):]:
        shutil.copy(fake, 'C:/ai/workspace/sh/newtrain/train/fake')
```

---



4) **Model Learn**

```python
import os
import pandas as pd
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.inception_resnet_v2 import InceptionResNetV2

os.environ["CUDA_VISIBLE_DEVICES"]= "1"

tf.debugging.set_log_device_placement(True)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

real_train_dir = './dataset/train/real/'
real_train = [real_train_dir + x for x in sorted(os.listdir(real_train_dir))]

fake_train_dir = './dataset/train/fake/'
fake_train = [fake_train_dir + x for x in sorted(os.listdir(fake_train_dir))]

real_test_dir = './dataset/test/real/'
real_test = [real_test_dir + x for x in sorted(os.listdir(real_test_dir))]

fake_test_dir = './dataset/test/fake/'
fake_test = [fake_test_dir + x for x in sorted(os.listdir(fake_test_dir))]

real_val_dir = './dataset/val/real/'
real_val = [real_val_dir + x for x in sorted(os.listdir(real_val_dir))]

fake_val_dir = './dataset/val/fake/'
fake_val = [fake_val_dir + x for x in sorted(os.listdir(fake_val_dir))]

train_path = real_train + fake_train
train_label = ['0']*len(real_train) + ['1']*len(fake_train)
train_df = pd.DataFrame({'path': train_path, 'label':train_label})
train_df = shuffle(train_df)

train_df.head()

test_path = real_test + fake_test
test_label = ['0']*len(real_test) + ['1']*len(fake_test)
test_df = pd.DataFrame({'path': test_path, 'label':test_label})
test_df = shuffle(test_df)

test_df.head()

val_path = real_val + fake_val
val_label = ['0']*len(real_val) + ['1']*len(fake_val)
val_df = pd.DataFrame({'path': val_path, 'label':val_label})
val_df = shuffle(val_df)

val_df.head()

# Data Augmentation
# rotation_range : 회전 최대 반경
# width_shift_range : 좌우 이동 최대 이미지 가로 사이즈
# height_shift_range : 상하 이동 최대 이미지 세로 사이즈
# horizontal_flip : 좌우 반전 실행
# vertical_flip : 상하 반전 실행
# rescale : 원본은 0-255의 RGB 계수로 구성되는데, 이는 모델을 효과적으로 학습시키기에 너무 높습니다 (통상적인 learning rate를 사용할 경우). 그래서 이를 1/255로 스케일링하여 0-1 범위로 변환시켜줍니다.

train_datagen = ImageDataGenerator(rotation_range = 270, width_shift_range = .2,
                                  height_shift_range = .2, horizontal_flip = True,
                                  vertical_flip = True, rescale = 1/255)

test_datagen = ImageDataGenerator(rescale = 1/255)

val_datagen = ImageDataGenerator(rescale = 1/255)

# Image Read
# flow_from_dataframe() : 디렉토리에서 이미지를 읽을 객체 생성
# train_df : 데이터 프레임
# X_col, y_col : 컬럼 지정
# batch_size : 한번에 리턴 할 이미지의 개수
# class_mode : 분류 방법 (binary : 이진 분류, categorical : 다중 분류)

IM_HEIGHT = 160
IM_WIDTH = 160

train_generator = train_datagen.flow_from_dataframe(train_df, x_col="path", y_col="label", target_size=(IM_HEIGHT, IM_WIDTH),
                                                   batch_size=16, class_mode='binary', shuffle=True)

test_generator = test_datagen.flow_from_dataframe(test_df, x_col="path", y_col="label", target_size=(IM_HEIGHT, IM_WIDTH),
                                                   batch_size=16, class_mode='binary', shuffle=True)

val_generator = val_datagen.flow_from_dataframe(val_df, x_col="path", y_col="label", target_size=(IM_HEIGHT, IM_WIDTH),
                                                   batch_size=16, class_mode='binary', shuffle=True)

# Create Model
# InceptionResNetV2 구조를 갖는 Model 생성
# Sequential() : 입력값을 읽어 예측을 할 Sequential 객체 생성
# add(IR) : InceptionResNetV2 대입
# Flatten() : 선형 회귀를 하기 위해 합성곱 연산을 수행한 결과를 1차원 배열 변환
# Dense : 선형 회귀를 수행할 객체
# Activation = 'relu' : 선형 회귀 후 relu 활성 함수 사용
# model.add(Dense(512, activation="relu")) : Dense 모델을 예측 할 수 있도록 model에 추가
# Dropout : Model Overfitting 방지
# Dense(1) : 출력 데이터 칸의 수는 1
# Activation = 'sigmoid' : 선형 회귀 후 sigmoid 함수를 활성 함수를 이용해 0, 1 값 리턴
# loss : 손실 함수는 모델을 컴파일하기 위해 필요한 변수 중 하나
# binary_crossentropy : 이항 교차 엔트로피(두 개의 클래스 중에서 예측할 때)
# optimizer=Adam(lr=.0001) : learning_rate를 0.0001로 설정
# model_path : 모델 저장 경로 지정
# EarlyStopping : 모델 학습 사전 중지 (monitor에서 정해준 기준이 개선 안될 시 중지)
# monitor : val_loss를 모니터링, 사전 중지 기준
# patience : 횟수를 주면 개선이 안돼도 횟수만큼 더 학습

IR = InceptionResNetV2(input_shape=(IM_HEIGHT, IM_WIDTH, 3),
                     include_top=False, weights='imagenet')

IR.summary()
model = Sequential()
model.add(IR)
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=.0001), metrics=['acc'])

model_path ="./model/IR-{epoch:02d}-{val_loss:.4f}.h5"
checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=30)

model.summary()

# fit_generator : 제너레이터로 이미지를 담고 있는 배치로 학습
# train_generator : 학습 데이터
# epochs : 학습 할 횟수
# validation_data : 학습 검증 데이터

history = model.fit_generator(train_generator, epochs=5000, validation_data = val_generator, callbacks=[checkpoint, early_stopping])

#Visualize the learning process

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss', 'acc', 'val_acc'], loc='upper left')
plt.show()
```

---



5) **Model Inference**

```python
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2 as cv
import os
from keras.models import load_model
from facenet_pytorch import MTCNN

os.environ["CUDA_VISIBLE_DEVICES"]= "1"

tf.debugging.set_log_device_placement(True)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Data Adopted & Data Read
# Kaggle Deepfake Detection Challenge Dataset의 https://www.kaggle.com/c/deepfake-detection-challenge/data
# 각각 약 1500개 영상의 REAL, FAKE 정보가 담긴, json 파일로 구성
# 각각 metadata를 Dataframe으로 읽어온 뒤, 하나의 Dataframe으로 구성
# metadata를 sort하여 알파벳 순으로 정렬

metadata0 = pd.read_json('/content/drive/My Drive/metadata/metadata0.json').T
metadata0 = pd.DataFrame(metadata0["label"])
metadata0.sort_index(inplace=True)

metadata1= pd.read_json('/content/drive/My Drive/metadata/metadata1.json').T
metadata1 = pd.DataFrame(metadata1["label"])
metadata1.sort_index(inplace=True)

metadata2 = pd.read_json('/content/drive/My Drive/metadata/metadata2.json').T
metadata2 = pd.DataFrame(metadata2["label"])
metadata2.sort_index(inplace=True)

metadata = pd.concat([metadata0, metadata1, metadata2])

metadata.sort_index(inplace=True)
metadata = metadata.reset_index()
metadata.columns=["filename", "label"]
metadata["label"]=np.where(metadata["label"]=="FAKE", 1, 0)

# Load Test Video

test = '/content/drive/My Drive/test_data/'
test_movie_files = [test + x for x in sorted(os.listdir(test))]

# Load Model

model = load_model("/content/drive/My Drive/IR-89-0.0000.h5")
model.summary()

# Predict Test Video
# v_cap : 영상을 프레임별로 자름
# v_len : 프레임의 총 개수를 변수에 저장
# cv2.VideoCapture.grab() : 프레임을 가져온다. 실패 시 success 변수엔 False가 저장
# cv2.VideoCapture.retrieve() : grab한 프레임을 decode한다. 프레임이 없을 경우 False가 success 변수에 저장
# cv.COLOR_BGR2RGB : 프레임을 BGR에서 RGB로 변환
# MTCNN : 얼굴을 검출하고, margin을 줘 얼굴이 살짝 잘리는 것을 방지
# count : 영상에서 프레임별로 검출되는 모든 얼굴에 대해 번호를 매기기 위함

%%time
detector = MTCNN(margin=50, keep_all=False, post_process=False, device='cuda:0',thresholds=[.9,.9,.9])
vid_num = 0
scores=[]
filenames = []

for vid in test_movie_files:
    predict_all=[]
    count=0
    file_name_mp4 = vid.split('/')[-1]
    file_name = file_name_mp4.split('.')[0]
    v_cap = cv.VideoCapture(vid)    
    v_len = int(v_cap.get(cv.CAP_PROP_FRAME_COUNT))
    for frm in range(v_len):    
        success = v_cap.grab()
        if frm % 1 == 0:
            success, frame = v_cap.retrieve()
            if not success:
                continue
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame = detector(frame)
            if frame is not None:
                frame = np.transpose(frame, (1, 2, 0))
                frame = np.array(cv.resize(np.array(frame),(160 ,160)))
                frame = (frame.flatten() / 255.0).reshape(-1, 160, 160, 3)
                count = count+frame.shape[0]
                predict = model.predict(frame)
                predict_all.append(predict)
            else:
                continue
        else:
            continue
    if (count>11):
        predict_all.sort()
        final_pred = sum(predict_all[5:-5])/(count-10)
        if final_pred == 1: final_pred = [[0.99]]
        elif final_pred == 0: final_pred = [[0.01]]
        scores.append(final_pred[0][0])
        print(" score :",final_pred[0][0]*100)
    else:
        scores.append(0.5)
        print(" score :",50)   
    filenames.append(file_name_mp4)
    print('filename :',file_name_mp4)
v_cap.release()

# Make Predict Dataframe
# acc를 확인하기 위해 0.5 이상인 값을 1로, 미만인 값을 0으로 변환
# metadata의 label과 df의 predict를 filename를 기준으로 merge

df = pd.DataFrame({'filename':filenames, 'predict':scores}) 
df["predict"]=np.where(df["predict"]>=0.5, 1, 0)

predict_df = pd.merge(metadata, df, on='filename')

compare = (predict_df.label == predict_df.predict)
compare = np.sum(compare)
acc = compare / len(predict.predict)
acc
```

---

<h3>3. 시연 영상</h3>

https://youtu.be/uKsX4jroHeA

---

<h3>4. 배운 것</h3>

1) MTCNN 라이브러리를 이용하여 얼굴 검출시 얼굴이 살짝 잘려 이목구비만으로 학습한 것 보단 Margin을 주어 얼굴이 원 형태를 유지하여 학습 하는 것이 최소 10% 이상 학습 효율을 높일 수 있었다.



2)  케라스에서 지원하는 모델들을 여러가지 사용해봤을 떄 InceptionResnetV2가 가장 성능이 좋았다. 성능이 좋은 많큼 모델 용량이 가장 컸다. 그에 반해 MobileNetV2와 NASNetMobile이 모델 용량이 적으면서 준수한 성능을 보여주었다. 추후에 기회가 된다면 MobileNetV2과 NASNetMobile을 이용하여 학습을 시키는 방향도 공부해 봐야겠다. 그리고 가장 중요한 것은 직접 만들지 못한 모델을 다음엔 직접 만들어 학습을 시켜봐야겠다.

![모델 성능](https://user-images.githubusercontent.com/57612261/106576181-38513f00-6580-11eb-9d0e-6ab9ec27677e.png)



3)  4가지의 모델들을 학습에 많은 시간이 소요 되었다.  GPU Geforce 1080 8GB로 학습을 하였는데 Out Of Memory(이하 OOM)가 자주 발생하여 구글 Colab Pro로 학습을 진행 하였다.  OOM이 발생하지 않는 법을 공부하고 케라스에서 지원하는 모델들은 메모리를 많이 필요하다는 것을 배웠다.
