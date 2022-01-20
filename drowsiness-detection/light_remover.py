import cv2

def light_removing(frame) :
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 흑백화
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB) # LAB 색공간으로 변환
    L = lab[:,:,0] # LAB 색공간 중 L 채널만 걷어내기
    med_L = cv2.medianBlur(L,99) # 미디안 블러링 : 무작위 노이즈를 제거한다. 99x99 내의 픽셀을 크기순으로 정렬 후 중간값을 뽑아서 픽셀값으로 사용(1보다 큰 홀수)
    invert_L = cv2.bitwise_not(med_L) # bitwise_not 색 반전
    composed = cv2.addWeighted(gray, 0.75, invert_L, 0.25, 0) # img1 * a +imgb * b + c 흑백화한 화면과 위 작업을 한 화면을 합침
    return L, composed

