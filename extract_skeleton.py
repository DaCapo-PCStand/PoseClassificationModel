# 필요한 관절값만 뽑는
import argparse
import cv2
import math
import numpy as np
import pandas as pd
import csv
import csv_control

#################################################################

def calculate_distance(aX, aY, bX, bY):
    return math.sqrt(math.pow(aX-bX,2)+math.pow(aY-bY,2))

def calculate_slope(aX,aY,bX,bY):
    if(aX-bX==0):
        return 0
    return abs(aY-bY)/abs(aX-bX)

def output_keypoints(l,frame, proto_file, weights_file, threshold, model_name, BODY_PARTS):
    global points

    check_flag = 0

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
        else:
            if(i==2)or(i==5)or(i==14)or(i==15):
                print(i, "가 감지되지 않았음")
                check_flag = 1
                continue

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
    
    dict = {"image_name":0,
            "right_shoulder_Y":RShoulderPoints[1],
            "left_shoulder_Y":LShoulderPoints[1],
            "shoulder_eye_distance":ShoulderEyeDistance/EyeDistance,
            "eye_distance":EyeDistance,
            "eye_slope":EyeSlope,
            "shoulder_slope":ShoulderSlope,
            "label":l}
    
    print(dict)
    #cv2.imshow("Output_Keypoints", frame)
    cv2.waitKey(0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    return dict, check_flag

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

ap = argparse.ArgumentParser()
ap.add_argument("--label", default = "zero")
ap.add_argument("--start", type=int)
ap.add_argument("--end", type=int)

ap.add_argument("--first", type=int, default =1)
args = vars(ap.parse_args())

path = "./data_"+args["label"]+"/"
csv_path = "./0417_data_"+args["label"]+'.csv'

img = path+"image_{}.jpg"

l=0
if args["label"] == "zero":
    l = 0
elif args["label"] == "one":
    l = 1
elif args["label"] == "two":
    l = 2

rows = []

for i in range(args["start"], args["end"]):
    image_name = img.format(i)
    frame_coco = cv2.imread(image_name)

    if(frame_coco is None):
        print(i," is not exist")
        continue

    # 키포인트를 저장할 빈 리스트
    points = []

    # COCO Model
    frame_COCO, flag = output_keypoints(l,frame=frame_coco, proto_file=protoFile_coco, weights_file=weightsFile_coco,
                             threshold=0.1, model_name="COCO", BODY_PARTS=BODY_PARTS_COCO)
    
    if(flag==1):
        continue
    else:
        frame_COCO['image_name'] = image_name

        rows.append(frame_COCO)
        print(i)
        print()
        #print(frame_COCO)
        if args['first']==1:
             csv_control.first_make_csv(frame_COCO, csv_path)

        elif args['first']==0:
             csv_control.add_to_csv(frame_COCO, csv_path)
