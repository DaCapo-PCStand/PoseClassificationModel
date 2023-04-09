import cv2
import time
import numpy as np
from random import randint
import argparse

# argparse는 커맨드 라인으로 인수를 받아 처리할 수 있게 하는 표준 라이브러리
# 1. argparse를 임포트
# 2. parser를 생성 
# 3. 인자 설정
# 4. 분석

# 인자값을 받을 수 있는 인스턴스 생성
parser = argparse.ArgumentParser(description='Run keypoint detection')
# 입력받을 인자값 등록
parser.add_argument("--device", default="cpu", help="Device to inference on")
parser.add_argument("--image_file", default="group.jpg", help="Input image")

# 입력받은 인자를 args에 저장
args = parser.parse_args()

image1 = cv2.imread(args.image_file)

# 모델을 구성하는 레이어들의 속성을 저장하고 있는 protoFile
# 훈련된 모델의 가중치를 저장하고 있는 caffemodel 로드
protoFile = "openpose_coco\\pose_deploy_linevec.prototxt"
weightsFile = "openpose_coco\\pose_iter_440000.caffemodel"

# 추출할 관절의 개수
nPoints = 18
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

# 관절들을 연결
POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16] ]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]

# 관절을 추출하는 함수
# 특정 관절이 위치할 확률인 probMap, 임계값이 인풋
def getKeypoints(probMap, threshold=0.1):

    # 입력영상:probmap
    # 가우시안 커널 크기:(3,3)
    # x, y방향 sigma:둘 다 0
    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

    # mapSmooth>임계값이면
    # uint8은 RGB에서 사용하는 타입으로 0~255
    mapMask = np.uint8(mapSmooth>threshold)

    # 추출한 관절을 저장할 행렬
    keypoints = []

    # find the blobs
    # 정확한 keypoint를 찾으려면 각 blob에 대한 최댓값을 찾아야 함
    # 1. keypoint에 해당하는 영역의 모든 윤곽선을 찾음
    # 2. 그 영역에 대한 mask 생성
    # 3. probMap에 mask를 곱해 그 영역에 대한 probMap 추출
    # -> 특정 인덱스의 관절일 가능성을 나타내는 probMap에서 
    # 4. 그 영역에 대한 로컬 최댓값을 찾음

    # 윤곽선을 검출하는 주된 요소는 흰색 객체를 검출함
    # 따라서 배경은 검게 하고, 검출하려는 물체는 흰색이 되도록 변형함 
    # 이미지의 윤곽선과 계층 구조를 반환함
    # 윤곽선은 Numpy 구조의 배열로 검출된 윤곽선의 지점이 담겨있음
    # 계층 구조는 각 윤곽선에 해당하는 속성 정보가 담겨있음

    # cv2.findContours(image, mode, method, contours=None, hierarchy=None, offset=None) -> contours, hierarchy
    # mode : 외곽선 검출 모드
    # method : 외곽선 근사화 방법
    # contours : 검출된 외곽선 좌표. numpy.ndarray로 구성된 리스트
    # hierarchy: 외곽선 계층 정보. numpy.ndarray. shape=(1, N, 4).
    # dtype=numpy.int32. hierarchy[0, i, 0] ~ hierarchy[0, i, 3]이 순서대로 next, prev, child, parent 외곽선 인덱스를 가리킴. 해당 외곽선이 없으면 -1.

    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #for each blob find the maxima
    for cnt in contours:
        # mapMask 크기의 영행렬 생성
        blobMask = np.zeros(mapMask.shape)
        # fillConvexPoly(img, pts, color)는 채워진 볼록 다각형을 그림
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)

        # probmap에 가우시안블러한 것과 관절이라고 추정되는 위치 곱셈
        maskedProbMap = mapSmooth * blobMask
        # 한 사람씩 각 keypoint의 xy좌표, probability score, ID 저장
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints

# Find valid connections between the different joints of a all persons present
# 각 관절의 keypoint를 찾았으니 이제 다른 관절과 이어야 함
# 예를 들어 코와 목을 연결할 때
# 무작정 1번 사람의 코와 가장 가까운 목이랑 연결하는 건 잘못될 가능성 있음
# 그래서 Part Affinity Maps이 필요한 것
# Part Affinity Maps : 최소거리랑 PAF에 따른 방향을 가짐

# PAF이 있어서 최소거리로 구한 pair가 맞지 않아도
# PAF will comply only with the unit vector joining
# Elbow and Wrist of the 2nd person.

# 이 라인에서의 n점을 찾음 이 점들이 같은 방향인지 PAF를 체크함
# 그 방향이 특정한 정도와 매치되면 맞는 페어임
def getValidPairs(output):
    # 맞는 페어
    valid_pairs = []
    # 잘못된 페어
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        # 코와 목을 잇는다?
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        # 페어 후보들
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # 만약 pair 찾기를 위한 키포인트가 감지되면
        # canA와 canB의 모든 관절을 체크함
        # 두 조인트 사이의 거리를 계산함
        # PAF를 찾음
        # 위의 공식을 이용해 connection valid를 표시하기 위한 스코어를 계산함

        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            # nA에 있는 모든 관절?
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                # nB에 있는 모든 관절?
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
            print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs


# 각 사람들의 키포인트 리스트를 생성
# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints


frameWidth = image1.shape[1]
frameHeight = image1.shape[0]

t = time.time()
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
if args.device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif args.device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

# Fix the input Height and get the width according to the Aspect Ratio
inHeight = 368
inWidth = int((inHeight/frameHeight)*frameWidth)

inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                          (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)


output = net.forward()
# 이 output은 4D Matrix임
# 1. The first dimension: image ID(네트워크에 두 개 이상의 이미지를 넣을 때)
# 2. The Second dimension: keypoint의 index를 나타냄
# 모델은 모든 사람의 관절 위치를 결정하는 Confidence map,
# 신체부위 사이의 연관 정도를 나타내는 Part Affinity Fields를 생성함
# COCO 모델은 57개의 parts(18개의 keypoint confidence maps+1background+19*2 part affinity maps)
# 3. The third dimension: output map의 height
# 4. The fourth dimension: output map의 width
print("Time Taken in forward pass = {}".format(time.time() - t))

detected_keypoints = []
keypoints_list = np.zeros((0,3))
keypoint_id = 0
threshold = 0.1

for part in range(nPoints):
    probMap = output[0,part,:,:]
    probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
    keypoints = getKeypoints(probMap, threshold)
    print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
    keypoints_with_id = []
    for i in range(len(keypoints)):
        keypoints_with_id.append(keypoints[i] + (keypoint_id,))
        keypoints_list = np.vstack([keypoints_list, keypoints[i]])
        keypoint_id += 1

    detected_keypoints.append(keypoints_with_id)


frameClone = image1.copy()
for i in range(nPoints):
    for j in range(len(detected_keypoints[i])):
        cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
cv2.imshow("Keypoints",frameClone)
                       
valid_pairs, invalid_pairs = getValidPairs(output)
personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)

for i in range(17):
    for n in range(len(personwiseKeypoints)):
        index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
        if -1 in index:
            continue
        B = np.int32(keypoints_list[index.astype(int), 0])
        A = np.int32(keypoints_list[index.astype(int), 1])
        cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)


cv2.imshow("Detected Pose" , frameClone)
cv2.waitKey(0)