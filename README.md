# 2025 Google-아주대학교 AI융합캡스톤디자인 대회 - 꼼 D-coA팀

라즈베리파이5 + Picamera2 + OpenCV(C++) 통합 프로젝트

# 1. 빌드
```
cd ~/D-COA/build
cmake ..
make
```
# 3. 실행
```
cd ~/D-COA
./streaming.sh
```

## 구현 완료
- Picamera2 &rarr; 공유 메모리 &rarr; OpenCV(C++) 통합
- 실시간 프레임 캡처 및 표시
- FPS 모니터링

## TODO
- AI 검증 (YOLO, 라이브니스, 딥페이크)
- FPGA 모듈 (PUF, 서명, 타임스탬프)
- 블록체인 연동