# 얼굴에서 점 5개만 추출
# 눈머리, 눈꼬리, 입
# 파일명 -p dlib_face_landmarks\shape_predictor_5_face_landmarks.dat

from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2

# 인자를 받을 수 있는 인스턴스 생성
ap = argparse.ArgumentParser()

# parser에 인자 추가시키기. 
# --shape-predictor는 runtime 때 로드되는
# face landmark predictor의 path를 변경할 수 있게 함
ap.add_argument("-p", "--shape-predictor", required=True,
	help="dlib_face_landmarks\shape_predictor_5_face_landmarks.dat")

# ap 객체의 인자들을 파싱함
args = vars(ap.parse_args())

# 이미 학습된 HOG + Linear SVM face detector를 초기화하고
# shape_predictor 파일을 로드함
print("[INFO] loading facial landmark predictor...")
# 얼굴 영역 탐지
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

cap = cv2.VideoCapture(1)

if cap.isOpened():
    while(True):
        # ret : 프레임 읽기 성공 여부 True/False
        # frame : 프레임 이미지, Numpy 배열 또는 None
        ret, frame = cap.read()    # Read 결과와 frame

        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # 흑백영상에서 얼굴을 감지했으면 좌표정보가 들어감
        rects = detector(gray, 0)
        
        # 얼굴 개수만큼 반복
        for rect in rects:
            # determine the facial landmarks for the face region, then                # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
        
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw each of them
            for (i, (x, y)) in enumerate(shape):
                # 이미지, 중심, 반지름, 색, 선 두께(-1은 다 채우라는 뜻)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    
        # ord('q')는 그에 맞는 숫자로 변환해줌
        
        if key == ord("q"):
            break
    # do a bit of cleanup
    cv2.destroyAllWindows()
    frame.stop()
            

else:
    print("Can't open Video")