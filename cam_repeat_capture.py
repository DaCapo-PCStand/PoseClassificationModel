# 반복해서 사진 찍는 코드
import argparse                                                                                                                                     
import cv2
from time import sleep

ap = argparse.ArgumentParser()
ap.add_argument("--label", default = "zero")
# 현재 사진 장수의 십의자리 -> 10장 단위로 찍게 코드를 짜서 편의상으로 이렇게 함
ap.add_argument("--cnt", type=int)

# ap 객체의 인자들을 파싱함
args = vars(ap.parse_args())

path = "./0524_data_"+args["label"]+"/"

filename = "image_{}.jpg"
i = 10*args["cnt"]

# 비디오 캡쳐 객체 생성 
cap = cv2.VideoCapture(0)

if cap.isOpened():
    while(True):
        ret, frame = cap.read()
        # 영상 크기를 640x480에서 축소함
        # interpolation=cv2.INTER_AREA 는 영상 축소할 때 효과적인 보간법
        frame=cv2.resize(frame,dsize=(320,240),interpolation=cv2.INTER_AREA)

        # 프레임 읽기를 성공했으면
        if(ret) :
            cv2.imshow('frame_color', frame)    # 컬러 화면 출력
            img_name = filename.format(i)
            cv2.imwrite(path+img_name,frame)
            i = i+1
            sleep(0.5)
            if(i%50==0):
                break
            if cv2.waitKey(1)==ord('q'): 
                break
        else:
            print('no frame')
            break
else:
    print("Can't open Video")

# 비디오 객체 자원 반납
cap.release()
# 윈도우창 삭제
cv2.destroyAllWindows()