# test_video_reader.py : OpenCV 연습장

import cv2
import time 

VIDEO_FILE_PATH = "my_test_video.mp4"

cap = cv2.VideoCapture(VIDEO_FILE_PATH)

if not cap.isOpened():
    print(f"오류: '{VIDEO_FILE_PATH}' 파일을 열 수 없습니다.")
    # (파일이 없거나, 이름이 틀렸거나, opencv가 영상을 못 읽는 경우)
else:
    print("영상 파일을 성공적으로 열었습니다! 1초에 한 프레임씩 재생합니다.")
    print("!!! 중요: 이 코드는 SSH 원격 접속 환경에서는 창이 뜨지 않습니다. !!!")

    while True:
        #    ret: 성공 여부 (True/False)
        #    frame: '숫자 판'으로 된 실제 이미지 데이터 (Numpy)
        ret, frame = cap.read()

        if not ret:
            print("영상이 끝났습니다. 루프를 종료합니다.")
            break 

        # 리사이즈 크기 설정 (가로 640, 세로 480)
        new_width = 640
        new_height = 480
        
        # 리사이즈 연습
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # 노이즈 제거 연습
        denoised_frame = cv2.GaussianBlur(resized_frame, (5, 5), 0)

        # CLAHE을 사용하기전에, 영상을 흑백으로 변환
        gray_frame = cv2.cvtColor(denoised_frame, cv2.COLOR_BGR2GRAY)

        # CLAHE 필터값 설정
        clahe_filter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # 흑백으로 처리한 영상에 CLAHE 필터 적용
        enhanced_frame = clahe_filter.apply(gray_frame)

        # ??컬러 이미지로 안바꿈? -> 밝기 (Y)와 색정보 (Cr, Cb)를 분리하여 Y만 조정하는 작업 필요.

        print(f"원본 크기: {frame.shape}, 리사이즈된 크기: {resized_frame.shape}, 노이즈 제거 후 크기: {denoised_frame.shape}, CLAHE(흑백): {enhanced_frame.shape}")
        
        # 1초에 한 프레임을 처리
        time.sleep(1) 

print("재생기(cap)를 닫습니다.")
cap.release()