# Tensorflow_Face_recognition_CNN
## Summary
CNN(Convolution Neural Network)를 사용하여 얼굴을 인식하여 분류하는 모델을 만들고자 한다.

메시, 손흥민, 호날두 3명을 분류하고자 한다.

이미지 데이터 셋을 직접 구하여 사용하는 것이 목적이다.


## Image crop
인물이 나온 사진을 다운 받기 위해 네이버에 검색후 'Fatkun Batch Download Image' 크롬 확장프로그램을 사용하여 한번에 쉽게 이미지를 다운받았다.

그 후 (crop_face.py 참조) OpenCV를 사용하여 사진에서 얼굴 부분을 찾아내어 100x100으로 저장하였다.

하지만 사진의 사이즈나 얼굴이 정확하지 않은 부분은 수작업으로 확인후 train과 eval로 나누어 저장하였다.

Trai : 175장

Eval: 25장

(인물별)



## Input

(input.py 참조)

Input.py에서는 Model이 호출하게 되면 로컬에 저장된 이미지를 4차원의 형태로 만들고 라벨링 작업을 한 후 넘겨주는 역할을 한다.



### Import

PIL, Numpy, OpenCV, os, random 라이브러리를 사용한다.

~~~python
from PIL import Image
import numpy as np
import cv2
import os
import random
~~~



### Input Function

data_set: ‘train’ or ‘eval’

batch_size: batch size

~~~python
def input(data_set, batch_size):
~~~



### Load Images

입력된 data_set의 이미지의 이름을 모두 불러온다.

(인물이 달라도 이름은 같기 때문에 호날두를 대표로 불러옴.)

~~~python
img_list = (os.listdir('Face_Image/' + data_set + '/ro'))
~~~



### Image resize and labeling

~~~python
people = ['messy', 'son', 'ro'] # 나중에 랜덤으로 뽑음.
img_batch = [] # 3차원의 이미지를 쑤셔 넣을 공간.
    Y_Labels = np.zeros((batch_size, 1)) # label 정보가 저장될 공간, 크기는 입력된 batch_size임
    # Y_Labels = np.zeros((batch_size, 3))
    for i in range(int(batch_size)): # 입력된 batch_size만큼 반복됨.
        people_random = random.choice(people) 메시, 손흥민, 호날두를 랜덤으로 뽑음.
        img = Image.open('Face_Image/' + data_set + '/' + people_random + '/' + str(random.choice(img_list))) # 입력된 data_set안에 랜덤으로 뽑은 사람의 사진중 랜덤으로 가져옴.
        img.thumbnail((64, 64)) # 100x100 사이즈를 64x64 사이즈로 압축
        # img_np = np.array(img) # 불러온 이미지를 np.array를 사용하여 [64, 64, 3] 배열로 변환
        tmp = random.randrange(1, 5) # 1~4 랜덤
        blur = cv2.blur(np.array(img), (int(tmp), int(tmp))) # 이미지에 1~4중 랜덤으로 노이즈를 줌.
        img_batch.append(np.array(blur)) # 3차원 이미지를 쑤셔 넣는다.

        if people_random == 'messy': # 처리한 이미지의 폴더가 messy일 경우 Label를 0으로.
            Y_Labels[i][0] = 0
        elif people_random == 'son': # 처리한 이미지의 폴더가 son일 경우 Label를 1으로.
            Y_Labels[i][0] = 1
        elif people_random == 'ro':# 처리한 이미지의 폴더가 ro일 경우 Label를 2으로.
            Y_Labels[i][0] = 2
        print(i) # 처리되고 있는 순서.
~~~



### convert 3d array to 4d array

~~~python
    result = np.concatenate([img_temp.reshape(1, 64, 64, 3) for img_temp in img_batch]) # 3차원 이미지를 [1, 64, 64, 3]으로 reshape한 후 result에 쑤셔 넣는다.

    return result, Y_Labels # model로 4차원 data_set과 2차원 Label을 반환한다.
~~~



## Full Sources

~~~python
from PIL import Image
import numpy as np
import cv2
import os
import random


def input(data_set, batch_size):


    img_list = (os.listdir('Face_Image/' + data_set + '/ro'))
    people = ['messy', 'son', 'ro']
    img_batch = []
    Y_Labels = np.zeros((batch_size, 1))
    # Y_Labels = np.zeros((batch_size, 3))
    for i in range(int(batch_size)):
        people_random = random.choice(people)
        img = Image.open('Face_Image/' + data_set + '/' + people_random + '/' + str(random.choice(img_list)))
        img.thumbnail((64, 64))
        # img_np = np.array(img)
        tmp = random.randrange(1, 5)
        blur = cv2.blur(np.array(img), (int(tmp), int(tmp)))
        img_batch.append(np.array(blur))

        if people_random == 'messy':
            Y_Labels[i][0] = 0
        elif people_random == 'son':
            Y_Labels[i][0] = 1
        elif people_random == 'ro':
            Y_Labels[i][0] = 2
        print(i)


    result = np.concatenate([img_temp.reshape(1, 64, 64, 3) for img_temp in img_batch])

    return result, Y_Labels
~~~





<br/>



## Model

Input: 64x64

Convolutional + Pooling layers: [5, 5, 3, 64]

Convolutional + Pooling layers: [5, 5, 64, 64]

Convolutional + Pooling layers: [4, 4, 64, 128]

Convolutional + Pooling layers: [4, 4, 128, 128]

Fully connected layers: [4 * 4 * 128, 512]

Fully connected layers: [512, 3]

Dropout layers: 0.8

Claassifier: use softmax





Next_batch: input에서 받아온 데이터 셋과 레이블을 shuffle 해준다.



Minibatch: 64

step 10회 반복 할때마다 eval 데이터 셋으로 테스트를 해본다.



~~~python
import input
import tensorflow as tf
import input
import numpy as np

def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def weight_variable(shape):
    initial = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.05))
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.05))
    return tf.Variable(initial)

def conv2d(x, w, bias):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + bias


def relu(x):
    return tf.nn.relu(x)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
Y_Label = tf.placeholder(tf.float32, shape=[None, 3])
keep_prob = tf.placeholder(tf.float32)


Conv1 = conv2d(X, weight_variable([5, 5, 3, 64]), bias_variable([64]))
Relu1 = relu(Conv1)
Pool1 = max_pool_2x2(Relu1)
# 32x32


Conv2 = conv2d(Pool1, weight_variable([5, 5, 64, 64]), bias_variable([64]))
Relu2 = relu(Conv2)
Pool2 = max_pool_2x2(Relu2)
# 16x16


Conv3 = conv2d(Pool2, weight_variable([4, 4, 64, 128]), bias_variable([128]))
Relu3 = relu(Conv3)
Pool3 = max_pool_2x2(Relu3)
# 8x8


Conv4 = conv2d(Pool3, weight_variable([4, 4, 128, 128]), bias_variable([128]))
Relu4 = relu(Conv4)
Pool4 = max_pool_2x2(Relu4)
# 4x4

w_fc1 = tf.Variable(tf.truncated_normal(shape=[4 * 4 * 128, 512], stddev=5e-2))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]))

h_conv5_flat = tf.reshape(Pool4, [-1, 4 * 4 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, w_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = tf.Variable(tf.truncated_normal(shape=[512, 3], stddev=5e-2))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[3]))
logits = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
y_pred = tf.nn.softmax(logits)

x_train, y_train = input.input('train', 5000)
x_test, y_test = input.input('eval', 10000)

y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 3), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 3), axis=1)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_Label, logits=logits))
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y_Label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # save_path = "./model/model_1.ckpt"
    # saver.restore(sess, save_path)

    for step in range(1000):
        batch = next_batch(64, x_train, y_train_one_hot.eval())

        if step % 10 == 0:
            test_batch = next_batch(1000, x_test, y_test_one_hot.eval())
            accuracy_print = accuracy.eval(feed_dict={X: test_batch[0], Y_Label: test_batch[1], keep_prob: 1.0})
            Loss_print = loss.eval(feed_dict={X: batch[0], Y_Label: batch[1], keep_prob: 1.0})
            print(step, accuracy_print, Loss_print)



        sess.run(train_step, feed_dict={X: batch[0], Y_Label: batch[1], keep_prob: 0.8})
~~~



## Result

990step까지 학습을 하였는데 가장 학습률이 높았던 때는 960 step이다.

91.5% 였다.

나쁘지 않은 결과인 것 같다..

~~~
0 0.345 197.6069
10 0.362 4.6653957
20 0.453 1.8727981
30 0.352 2.8304136
40 0.429 1.0768447
50 0.338 1.4988775
60 0.298 1.2436678
70 0.414 0.9301225
80 0.433 0.9754803
90 0.496 0.8383784
100 0.325 22.122532
110 0.345 4.7652674
120 0.332 2.0847955
130 0.428 0.851163
140 0.434 1.1544795
150 0.375 1.1966105
160 0.592 0.83420295
170 0.491 0.7996795
180 0.629 0.85008633
190 0.493 0.94538546
200 0.341 1.8116097
210 0.624 0.62232643
220 0.555 0.71296453
230 0.567 0.7229748
240 0.689 0.7708217
250 0.727 0.44554853
260 0.67 0.46262497
270 0.608 0.5948684
280 0.508 1.2609433
290 0.748 0.30215204
300 0.489 1.2131842
310 0.779 0.24752215
320 0.693 0.15010002
330 0.591 0.65155387
340 0.77 0.29027015
350 0.801 0.2599817
360 0.719 0.6305636
370 0.792 0.17317513
380 0.786 0.064622775
390 0.581 2.236622
400 0.799 0.19989356
410 0.841 0.088487566
420 0.879 0.10182723
430 0.825 0.011365123
440 0.647 0.54435813
450 0.824 0.024591135
460 0.817 0.06408346
470 0.699 0.18983218
480 0.847 0.02156632
490 0.822 0.016898733
500 0.788 0.10631196
510 0.706 0.027493387
520 0.801 0.009679989
530 0.567 0.9951723
540 0.819 0.017440798
550 0.835 0.0465378
560 0.83 0.0026295378
570 0.777 0.8934554
580 0.816 0.02697164
590 0.871 0.0057017067
600 0.838 0.20418634
610 0.856 0.0016530138
620 0.855 0.0005207339
630 0.824 0.7005123
640 0.862 0.06293791
650 0.868 0.013759151
660 0.845 0.0011848528
670 0.865 0.0045195795
680 0.818 0.48918718
690 0.861 0.009853948
700 0.858 0.002669801
710 0.853 0.00017557593
720 0.861 0.00020319465
730 0.862 4.5616736e-05
740 0.87 1.0722255e-05
750 0.873 8.4355874e-05
760 0.523 2.181731
770 0.87 0.018477365
780 0.867 0.015096102
790 0.85 0.005736928
800 0.851 0.00094226695
810 0.826 0.10626664
820 0.862 0.00071934325
830 0.873 0.0019402205
840 0.851 0.00016183215
850 0.858 0.0065656835
860 0.878 1.7194852e-05
870 0.849 0.2754978
880 0.899 0.0024340826
890 0.889 2.105256e-05
900 0.911 3.670228e-05
910 0.76 0.46754885
920 0.81 0.008710145
930 0.871 0.0004285363
940 0.892 0.0023724951
950 0.676 0.65665865
960 0.915 0.048428793
970 0.914 0.024017924
980 0.907 0.00046830025
990 0.665 0.84971166
~~~

