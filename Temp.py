from PIL import Image
import numpy as np
import tensorflow as tf
import os
import random


def input(data_set, batch_size):
    img_list = (os.listdir('Face_Image/' + data_set + '/비와이'))
    people = ['비와이', '손흥민', '트럼프']
    img_batch = []
    Y_Labels = np.zeros((batch_size, 3))
    print(Y_Labels)
    for i in range(int(batch_size)):
        people_random = random.choice(people)
        img = Image.open('Face_Image/' + data_set + '/' + people + '/' + str(random.choice(img_list)))
        img_np = np.array(img)
        img_batch.append(img_np)


        if people == '비와이':
            Y_Labels[i][0] = 1
        elif people == '손흥민':
            Y_Labels[i][1] = 1
        elif people == '트럼프':
            Y_Labels[i][2] = 1

    result = np.concatenate([img_temp.reshape(1, 64, 64, 3) for img_temp in img_batch])
    return result, Y_Labels