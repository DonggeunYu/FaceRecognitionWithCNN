# Tensorflow_Face_recognition_CNN
## Summary
CNN(Convolution Neural Network)를 사용하여 얼굴을 인식하여 분류하는 모델을 만들고자 한다.
구글 클라우드 API를 사용하여 얼굴 데이터 셋을 작업 할 계획이다.

## Use Python Library

Google Cloud Vision API
~~~
pip install --upgrade google-api-python-client
~~~

oauth2client
~~~
pip install --upgrade oauth2client
~~~

PIL
~~~
pip install image
~~~

## Python File
crop_face.py: 얼굴 사진 크로핑