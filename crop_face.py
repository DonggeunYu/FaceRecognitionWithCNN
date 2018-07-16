import cv2

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

        cropped = img[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
        # 이미지를 저장
        cnt += 1
        cv2.imwrite('Face_Image/' + people + '/' + str(cnt) + '.png', cropped)

    cv2.imshow('Image view', img)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    min += 1

    if min == max:
        return

    crop(min, max, cnt, people)




a = int(input())
b = str(input())
crop(1, a, 0, b)