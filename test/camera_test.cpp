#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cstring>
#include <memory>
#include "shared_memory.h"

int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "  C++ 프레임 처리 프로세스" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    const int FRAME_WIDTH = 640;
    const int FRAME_HEIGHT = 640;
    const size_t FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 3;

    std::cout << "공유 메모리 연결 대기 중..." << std::endl;

    std::unique_ptr<SharedMemory> shm;

    // 최대 10초 대기
    for (int i = 0; i < 100; i++)
    {
        try
        {
            shm.reset(new SharedMemory("frame_buffer", FRAME_SIZE, false));
            if (shm->isValid())
                break;
            shm.reset();
        }
        catch (...)
        {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (!shm || !shm->isValid())
    {
        std::cerr << "공유 메모리 연결 실패" << std::endl;
        std::cerr << "Python 프로세스가 먼저 실행되어야 합니다" << std::endl;
        return -1;
    }

    std::cout << "공유 메모리 연결 성공" << std::endl;
    std::cout << "프레임 처리 시작" << std::endl;
    std::cout << "웹 브라우저 확인: http://192.168.0.34:5000" << std::endl;
    std::cout << std::endl;

    cv::Mat frame(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3);
    int frame_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    // TODO: AI 검증, FPGA 처리 등 추가 예정

    while (true)
    {
        // 공유 메모리에서 프레임 읽기 (RGB 형식)
        std::memcpy(frame.data, shm->getBuffer(), FRAME_SIZE);
        frame_count++;

        // FPS 계산 및 출력
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           current_time - start_time)
                           .count();

        if (elapsed >= 5000)
        {
            double fps = frame_count / (elapsed / 1000.0);
            std::cout << "처리 FPS: " << static_cast<int>(fps)
                      << " | 프레임: " << frame_count << std::endl;
            frame_count = 0;
            start_time = current_time;
        }

        // === TODO: 프레임 처리 로직 ===
        // 1. AI 검증
        //    - 객체 탐지 (YOLO/MobileNet)
        //    - 라이브니스 검증 (얼굴 랜드마크)
        //    - 딥페이크 탐지 (픽셀 패턴, 압축 아티팩트)
        //
        // 2. FPGA 모듈 연동
        //    - PUF 디바이스 인증
        //    - SHA-256 해시 생성
        //    - ECC 디지털 서명
        //    - RTC/GPS 타임스탬프
        //
        // 3. 메타데이터 생성 및 블록체인 기록

        // 예시 코드:
        // cv::Mat bgr_frame;
        // cv::cvtColor(frame, bgr_frame, cv::COLOR_RGB2BGR);
        // bool verified = ai_detector.verify(bgr_frame);
        // if (verified) {
        //     auto metadata = fpga.generateMetadata(frame.data, FRAME_SIZE);
        //     blockchain.record(metadata);
        // }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << "\n종료 중..." << std::endl;
    std::cout << "처리 완료" << std::endl;

    return 0;
}