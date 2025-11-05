import cv2
import time 

VIDEO_FILE_PATH = "my_test_video.mp4"
# 1. 테스트 목적에 맞게 target_size를 직접 정의합니다 (640x480으로 테스트).
TARGET_SIZE = (640, 480) # (너비, 높이) 순서입니다. 
frame_number = 0 # 프레임 번호를 세기 위한 변수

cap = cv2.VideoCapture(VIDEO_FILE_PATH)

if not cap.isOpened():
    # 생략하지 않고 오류 처리 코드를 모두 포함합니다.
    print(f"오류: '{VIDEO_FILE_PATH}' 파일을 열 수 없습니다.")
    print("파일 경로 또는 파일명이 올바른지 확인해 주세요.")
    sys.exit(1) # 오류가 발생했으므로 프로그램 종료
else:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print("영상 파일을 성공적으로 열었습니다! 1초에 한 프레임씩 재생 및 전처리합니다.")
    print(f"-> 전처리: 크기를 {TARGET_SIZE}로 조정하고 BGR을 RGB로 변환합니다.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("영상이 끝났습니다. 루프를 종료합니다.")
            break 
        
        # 1. 프레임 번호 증가
        frame_number += 1
        
        if not ret or (total_frames > 0 and frame_number > total_frames):
            print(f"영상이 끝났습니다. (총 {frame_number - 1} 프레임 처리) 루프를 종료합니다.")
            break

        # 2. Preprocessor의 전처리 로직 (Resize + Color Convert)
        try:
            # 2-1. 크기 조정 (Resize)
            # INTER_AREA는 이미지를 축소할 때 성능과 품질이 좋습니다.
            resized_frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            
            # 2-2. 색상 변환 (Color Conversion)
            processed_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
        except cv2.error as e:
            print(f"오류: 전처리(Resize/Color Convert) 실패: {e}")
            break # 오류가 났으니 루프를 종료합니다.

        # 3. 처리될 때마다 진행 상황 출력
        print(f"프레임 #{frame_number} 처리 완료. (크기: {processed_frame.shape[0]}x{processed_frame.shape[1]})")

        # 4. 1초 대기 (1초마다 프레임을 처리하도록 만듭니다.)
        time.sleep(1.0) 

print("재생기(cap)를 닫습니다.")
cap.release()
# cv2.destroyAllWindows()sy