import cv2
# cv2는 색깔이 BGR

def output_keypoints(frame, proto_file, weights_file, threshold, model_name, BODY_PARTS):
    global points

    ### Step 1. 모델 다운로드

    ### Step 2. Load Network
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
    # 백엔드로 쿠다를 사용해서 속도향상을 꾀할 수 있다는데? 이거 어케하는거야

    # 입력 이미지의 사이즈 정의
    image_height = 320
    image_width = 240

    ### Step 3.Read Image and Prepare Input to the Network
    # 인풋이미지를 OpenCV로 읽으려면 input blob으로 변환시켜줘야 함
    # 그래야 네트워크에 넣을 수 있음
    # 이미지를 0~1사이 값으로 정규화해야 함 -> 그래서 1.0 / 255를 하는 것임
    # 이미지의 차원을 지정specify해야 함
    # 입력 영상에서 뺼 mean substraction 값 (0,0,0)
    # R과 B채널을 swap할 필요는 없음
    input_blob = cv2.dnn.blobFromImage(frame,
                                        1.0 / 255,
                                          (image_width, image_height),
                                            (0, 0, 0),
                                              swapRB=False, crop=False)
    # 전처리된 blob 네트워크에 입력
    net.setInput(input_blob)

    ### Step 4. Make Predictions and Parse KeyPoints
    # 네트워크의 출력을 얻기 위해 포워드 패스를 수행함
    # keypoint의 confidence map의 maxima를 찾아 keypoint의 위치를 찾음
    # reduce false detections로 threshold
    # keypoint가 detect되면 
    out = net.forward()

    # 이 output은 4D Matrix임
    # 1. The first dimension: image ID(네트워크에 두 개 이상의 이미지를 넣을 때)
    # 2. The Second dimension: keypoint의 index를 나타냄 
    # 모델은 모든 사람의 관절 위치를 결정하는 Confidence map,
    # 신체부위 사이의 연관 정도를 나타내는 Part Affinity Fields를 생성함
    # COCO 모델은 57개의 parts(18개의 keypoint confidence maps+1background+19*2 part affinity maps)
    # 3. The third dimension: output map의 height
    # 4. The fourth dimension: output map의 width

    
    # output map의 Height, width
    out_height = out.shape[2]
    out_width = out.shape[3]

    # 원본 이미지의 높이, 너비를 받아오기
    frame_height, frame_width = frame.shape[:2]

    # 포인트 리스트 초기화
    # 포인트의 (x,y)값을 저장하고 있음
    points = []

    print(f"\n============================== {model_name} Model ==============================")
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
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)

            points.append((x, y))
            print(f"[pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

        else:  # [not pointed]
            #cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            #cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)

            points.append(None)
            print(f"[not pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

    cv2.imshow("Output_Keypoints", frame)
    cv2.waitKey(0)

    t,_ = net.getPerfProfile()
    freq = cv2.getTickFrequency()/1000
    cv2.putText(frame, '%.2fms'%(t/freq), (10,20), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0,0,0))

    return frame

def output_keypoints_with_lines(frame, POSE_PAIRS):
    print()
    # 모든 연결부에 대해서
    for pair in POSE_PAIRS:
        part_a = pair[0]  # 0 (Head)
        part_b = pair[1]  # 1 (Neck)
        # 두 점이 모두 검출이 됐다면 선을 긋는다
        if points[part_a] and points[part_b]:
            print(f"[linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")
            cv2.line(frame, points[part_a], points[part_b], (0, 255, 0), 3)
        else:
            print(f"[not linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")


    cv2.rectangle(frame, (0,230),(640,290),(0,255,0), thickness=1, lineType=cv2.LINE_AA)
    cv2.imshow("output_keypoints_with_lines", frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


BODY_PARTS_COCO = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                   5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                   10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
                   15: "LEye", 16: "REar", 17: "LEar", 18: "Background"}

POSE_PAIRS_COCO = [[0, 1], [0, 14], [0, 15], [1, 2], [1, 5], [1, 8], [1, 11], [2, 3], [3, 4],
                   [5, 6], [6, 7], [8, 9], [9, 10], [12, 13], [11, 12], [14, 16], [15, 17]]


# 신경 네트워크의 구조를 지정하는 prototxt 파일 (다양한 계층이 배열되는 방법 등)
# 모델을 구성하는 레이어들의 속성을 저장하고 있음
protoFile_coco = "openpose_coco\\pose_deploy_linevec.prototxt"
# 훈련된 모델의 weight 를 저장하는 caffemodel 파일
weightsFile_coco = "openpose_coco\\pose_iter_440000.caffemodel"

# 이미지 경로
man = "images\\soccer.jpg"
# 이미지 읽어오기
frame_coco = cv2.imread(man)

# 키포인트를 저장할 빈 리스트
points = []

# COCO Model
frame_COCO = output_keypoints(frame=frame_coco, proto_file=protoFile_coco, weights_file=weightsFile_coco,
                             threshold=0.1, model_name="COCO", BODY_PARTS=BODY_PARTS_COCO)
output_keypoints_with_lines(frame=frame_COCO, POSE_PAIRS=POSE_PAIRS_COCO)

if(points[14]!=None and points[15]!=None):
    yEye = points[14][1]+points[15][1]
    print("points[14][1] : ", points[14][1])
    print("points[15][1] : ", points[15][1])
    print("yEye : ", yEye/2)

