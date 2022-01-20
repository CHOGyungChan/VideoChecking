import imutils
import time
import timeit
import dlib
import cv2
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import light_remover as lr
import ringing_sound as sound

def eye_aspect_ratio(eye) :     # EAR = Eye Aspect Ratio = 눈 가로 세로 비율
    A = dist.euclidean(eye[1], eye[5]) # 왼쪽 상하
    B = dist.euclidean(eye[2], eye[4]) # 오른쪽 상하
    C = dist.euclidean(eye[0], eye[3]) # 눈 좌우
    ear = (A + B) / (2.0 * C) # EAR 계산
    return ear

def init_open_ear() :
    time.sleep(5)
    print("open init time sleep")
    print("눈을 떴을 때의 크기 측정")
    ear_list = []
    th_message1 = Thread(target = init_message)
    th_message1.deamon = True
    th_message1.start()
    for i in range(7) : # 7번 반복
        ear_list.append(both_ear) # both_ear를 ear_list 마지막에 추가
        time.sleep(1)
    global OPEN_EAR
    OPEN_EAR = sum(ear_list) / len(ear_list) # 평균 뜬 눈 가로 세로 비율
    print("open list =", ear_list, "\nOPEN_EAR =", OPEN_EAR, "\n")

def init_close_ear() :
    time.sleep(2)
    th_open.join()
    time.sleep(5)
    print("close init time sleep")
    print("눈을 감았을 때의 크기 측정")
    ear_list = []
    th_message2 = Thread(target = init_message)
    th_message2.deamon = True
    th_message2.start()
    time.sleep(1)
    for i in range(7) :
        ear_list.append(both_ear)
        time.sleep(1)
    CLOSE_EAR = sum(ear_list) / len(ear_list)
    global EAR_THRESH
    EAR_THRESH = (((OPEN_EAR - CLOSE_EAR) / 2
                   ) + CLOSE_EAR) #EAR_THRESH 눈 뜬 상태의 50% (수치 조절 가능)
    print("close list =", ear_list, "\nCLOSE_EAR =", CLOSE_EAR, "\n")
    print("The last EAR_THRESH's value :",EAR_THRESH, "\n")

def init_message() :
    print("init_message")
    sound.sound("init_sound.mp3")

#####################################################################################################################

#1.
OPEN_EAR = 0
EAR_THRESH = 0

#2.
EAR_CONSEC_FRAMES = 40 # 0.1초 당 1.27
COUNTER = 1 #Frames counter.

#3.

closed_eyes_time = []
TIMER_FLAG = False
ALARM_FLAG = False

#4.
SLEEP_COUNT = 0
RUNNING_TIME = 0

#5.
PREV_TERM = 0

#6.
print("loading facial landmark predictor...")
print("얼굴 랜드마크 예측 변수 로드 중...")
detector = dlib.get_frontal_face_detector() # 얼굴 검출
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # shape_predictor_68_face_landmarks.dat(얼굴 68랜드마크) 불러오기

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] # face_utils.FACIAL_LANDMARKS_IDXS 에서 왼쪽 눈 불러오기
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"] # face_utils.FACIAL_LANDMARKS_IDXS 에서 오른쪽 눈 불러오기

#7.
print("starting video stream thread...")
print("비디오 스트림 시작 중...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

#8.
th_open = Thread(target = init_open_ear)
th_open.deamon = True
th_open.start()
th_close = Thread(target = init_close_ear)
th_close.deamon = True
th_close.start()

#####################################################################################################################

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width = 1000) # 캠 크기 조절

    L, gray = lr.light_removing(frame) # L, gray를 가져옴

    rects = detector(gray,0) # 흑백화 화면에서 얼굴검출

    for rect in rects: # 얼굴검출 무한반복
        shape = predictor(gray, rect) # 얼굴의 68 랜드마크를 알아낸다.
        shape = face_utils.shape_to_np(shape) # 얼굴의 랜드마크(x,y) 좌표를 np로 변환한다.

        leftEye = shape[lStart:lEnd] # LStart = 42 LEnd = 48
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye) # 위의 leftEye와 rightEye의 배열을 EAR 계산식에 대입
        rightEAR = eye_aspect_ratio(rightEye)

        #(leftEAR + rightEAR) / 2 => both_ear.
        both_ear = (leftEAR + rightEAR) * 500  #I multiplied by 1000 to enlarge the scope.

        leftEyeHull = cv2.convexHull(leftEye) # 윤곽선에서 블록 껍질을 검출한다.
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (217,178,255), 2) # 검출된 윤곽선을 그립니다.
        cv2.drawContours(frame, [rightEyeHull], -1, (217,178,255), 2) # (이미지, [윤곽선], 윤곽선 인덱스,(B,G,R), 두께, 선형타입)



        if both_ear < EAR_THRESH : # 졸고 있다고 판단
            if not TIMER_FLAG: # TIMER_FLAG가 FALSE 아래 실행
                start_closing = timeit.default_timer() # 시간 측정
                TIMER_FLAG = True
            COUNTER += 1

            if COUNTER >= EAR_CONSEC_FRAMES: # 졸고 있다고 판단 후 카운터가 EAR_CONSEC_FRAMES 이상이 됨

                mid_closing = timeit.default_timer() # mid_closing 시간 측정
                closing_time = round((mid_closing-start_closing)) # mid_closing-눈을 감은 때 소수점 3번째 까지 표시 = closing_time
                # closing_time = 감지 시간

                if closing_time >= RUNNING_TIME: # 감지 시간이 지남
                    if RUNNING_TIME == 0 :
                        CUR_TERM = timeit.default_timer() # 졸은 순간 + closing time = mid_closing = CUR_TERM
                        OPENED_EYES_TIME = round((CUR_TERM - PREV_TERM),3)
                        PREV_TERM = CUR_TERM
                        RUNNING_TIME = 5

                    RUNNING_TIME += 5 # 5초 간격으로 알림
                    ALARM_FLAG = True

                    print("졸음 경고!")
                    print("눈을 ", closing_time,"초 동안 감았습니다.")

        else : # both_ear > EAR_THRESH 눈을 떴다고 판단
            COUNTER = 0
            TIMER_FLAG = False
            RUNNING_TIME = 0

            if ALARM_FLAG :
                end_closing = timeit.default_timer()
                closed_eyes_time.append(round((end_closing-start_closing),3)) # 눈 뜬 시간 - 눈 감은 시간 = 눈을 감고있던 시간
                print("눈을 감고있던 시간:", round((end_closing-start_closing),3))
                SLEEP_COUNT += 1 # 졸음 수 카운팅
                print("{0}번째 졸았습니다.".format(SLEEP_COUNT))

            ALARM_FLAG = False

        cv2.putText(frame, "EAR : {:.2f}".format(both_ear), (300,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)

    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27: # 종료
        sleep_time = sum(closed_eyes_time)
        program_End = timeit.default_timer()
        print("#####################################")
        print("총 {0}번 졸았습니다.".format(SLEEP_COUNT))
        print("프로그램 동작시간 = ", round(program_End,3))
        print("졸음 감지 시간",round(sleep_time,3))
        print("수업 집중도 : ",100-round(sleep_time/program_End*100,),"%")
        break


cv2.destroyAllWindows()
vs.stop()

