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