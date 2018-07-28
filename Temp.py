from PIL import Image
import numpy as np
import tensorflow as tf
import os
import random


def input(data_set, batch_size):
    img_list = (os.listdir('Face_Image/' + data_set + '/비와이'))
    people = ['비와이', '손흥민', '트럼프']
    img_batch = []

    for i in range(int(batch_size)):
        img = Image.open('Face_Image/' + data_set + '/' + str(random.choice(people)) + '/' + str(random.choice(img_list)))
        img_np = np.array(img)
        img_batch.append(img_np)

    result = np.concatenate([img_temp.reshape(1, 64, 64, 3) for img_temp in img_batch])
    print(result.shape)
input('train', 10)