import cv2
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_casecade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')


def crop(min, max, cnt, people):
    print(min, cnt)
    img = cv2.imread('Image_Data/' + people + '/' + str(min) + '.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        if y - int(h / 4) < 1 or y + h + int(h / 4) < 1 or x - int(w / 4) < 1 or x + w + int(w / 4) < 1:
            break
        else:
            cropped = img[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
            #이미지를 저장
            cnt += 1
            cv2.imwrite('Face_Image/' + people + '/' + str(cnt) + '.png', cropped)
    print(img.shape)
    cv2.imshow('Image view', img)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    img_resize = Image.open('Face_Image/' + people + '/' + str(cnt) + '.png')
    img_resize.thumbnail((100, 100))
    img_resize.save('Face_Image/' + people + '/' + str(cnt) + '.png')
    min += 1

    if min == max:
        return

    crop(min, max, cnt, people)




print('start crop_face.py')
print('Image Size 100x100')
a = int(input('사진의 개수를 입력해주세요: '))
b = str(input('사람 이름을 입력해주세요: '))
crop(1, a+1, 0, b)