# 필요한 관절값만 뽑는
import argparse
import cv2
import math
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model



def getPosture(arr):
    # 우리가 만든 자세판별 모델을 로드
    model = load_model("models\\first_test_model.h5")
    # 모델에 입력값을 넣고 결과값 반환
    # 0, 1, 2 각 경우일 가능성을 numpy로 반환할걸?아마두
    arr2 = model.predict(np.expand_dims(arr, axis = 0))
    # 그 중 가능성이 가장 높은 것이 판단 결과임
    print(arr2)
    print(np.argmax(arr2))


#################################################################

# 두 점 사이의 거리 구하는 함수
def calculate_distance(aX, aY, bX, bY):
    return math.sqrt(math.pow(aX-bX,2)+math.pow(aY-bY,2))

# 두 점 사이의 기울기 구하는 함수
def calculate_slope(aX,aY,bX,bY):
    if(aX-bX==0):
        return 0
    return abs(aY-bY)/abs(aX-bX)

# 우리에게 필요한 뼈대값만 추출하는 함수
def output_keypoints(frame, proto_file, weights_file, threshold, model_name, BODY_PARTS):
    global points

    # 네트워크 불러오기
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    # 입력 사이즈 정의
    image_height = 320
    image_width = 240

    # 네트워크에 넣기 위한 전처리
    input_blob = cv2.dnn.blobFromImage(frame,1.0 / 255,(image_width, image_height),(0, 0, 0),swapRB=False, crop=False)

    # 전처리한 blob을 네트워크에 입력
    net.setInput(input_blob)

    # 결과 받아오고 높이, 너비 받아옴
    out = net.forward()
    out_height = out.shape[2]
    out_width = out.shape[3]

    # 원본 이미지의 높이, 너비를 받아옴
    frame_height, frame_width = frame.shape[:2]

    # 포인트 리스트 초기화
    # 포인트의 (x,y)값을 저장하고 있음
    points = []

    RShoulderPoints = [0,0]
    LShoulderPoints = [0,0]
    REyePoints = [0,0]
    LEyePoints = [0,0]

    for i in range(len(BODY_PARTS)):

        # 신체 부위의 confidence map
        prob_map = out[0, i, :, :]

        # 최소값, 최대값, 최소값 위치, 최대값 위치
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        # 원본 이미지에 맞게 포인트 위치 조정 -> 전처리과정에서 이미지 크기를 변경했기 때문
        x = (frame_width * point[0]) / out_width
        x = int(x)
        y = (frame_height * point[1]) / out_height
        y = int(y)

        # threshold가 작아지면 검출이 잘되고 그대신 오인식할 가능성도 높아짐
        if prob > threshold:  # [pointed] 감지했다는 뜻
            points.append((x, y))
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED) # circle(그릴곳, 원의 중심, 반지름, 색)
            cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            
            # 어깨 좌표 추출
            if i == 2:
                RShoulderPoints[0] = x
                RShoulderPoints[1] = y
            elif i == 5:
                LShoulderPoints[0] = x
                LShoulderPoints[1] = y
            # 눈 좌표 추출
            elif i == 14:
                REyePoints[0] = x
                REyePoints[1] = y
            elif i == 15:
                LEyePoints[0] = x
                LEyePoints[1] = y

    # 눈 ~ 어깨 거리. 광대 길이 대신에 눈 거리로 나누기
    ShoulderEyeDistance = calculate_distance((RShoulderPoints[0]+LShoulderPoints[0])/2, (RShoulderPoints[1]+LShoulderPoints[1])/2,
                            (REyePoints[0]+LEyePoints[0])/2, (REyePoints[1]+LEyePoints[1])/2)

    EyeDistance = abs(REyePoints[0]-LEyePoints[0])

    # 어깨기울기
    EyeSlope = calculate_slope(REyePoints[0], REyePoints[1], LEyePoints[0], LEyePoints[1])
    ShoulderSlope = calculate_slope(RShoulderPoints[0],RShoulderPoints[1],LShoulderPoints[0],LShoulderPoints[1])
    
    # 눈 ~ 어깨 거리 / 눈x거리
    # 어깨 기울기
    # 눈 기울기
    
    pointarr = [ShoulderEyeDistance/EyeDistance, EyeDistance, EyeSlope,ShoulderSlope]
    
    cv2.waitKey(0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    return pointarr

########################################

BODY_PARTS_COCO = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                   5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                   10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
                   15: "LEye", 16: "REar", 17: "LEar", 18: "Background"}

POSE_PAIRS_COCO = [[0, 1], [0, 14], [0, 15], [1, 2], [1, 5], [1, 8], [1, 11], [2, 3], [3, 4],
                   [5, 6], [6, 7], [8, 9], [9, 10], [12, 13], [11, 12], [14, 16], [15, 17]]

# 모델의 속성, 가중치가 저장된 경로
protoFile_coco = "openpose_coco\\pose_deploy_linevec.prototxt"
weightsFile_coco = "openpose_coco\\pose_iter_440000.caffemodel"

##############################################################

# 테스트할 이미지 경로
img = "images\\image_0.jpg"

# 테스트할 이미지 불러오기
frame_coco = cv2.imread(img)

# COCO Model
frame_COCO = output_keypoints(frame=frame_coco, proto_file=protoFile_coco, weights_file=weightsFile_coco,
                             threshold=0.1, model_name="COCO", BODY_PARTS=BODY_PARTS_COCO)

# 추출한 뼈대값을 모델에 넣고 결과값 얻어냄
getPosture(frame_COCO)