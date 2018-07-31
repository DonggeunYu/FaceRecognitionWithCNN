from PIL import Image
import numpy as np
import cv2
import os
import random


def input(data_set, batch_size):


    img_list = (os.listdir('Face_Image/' + data_set + '/호날두'))
    people = ['메시', '손흥민', '호날두']
    img_batch = []
    Y_Labels = np.zeros((batch_size, 3))
    for i in range(int(batch_size)):
        people_random = random.choice(people)
        img = Image.open('Face_Image/' + data_set + '/' + people_random + '/' + str(random.choice(img_list)))
        img.thumbnail((64, 64))
        img_np = np.array(img)
        blur = cv2.blur(np.array(img), (3, 3))
        img_batch.append(np.array(blur))

        if people == '메시':
            Y_Labels[i][0] = 1
        elif people == '손흥민':
            Y_Labels[i][1] = 1
        elif people == '호날두':
            Y_Labels[i][2] = 1

    result = np.concatenate([img_temp.reshape(1, 64, 64, 3) for img_temp in img_batch])
    print(np.shape(result))
    return result, Y_Labels