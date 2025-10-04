#pragma once
#include <string>
#include <cstddef>

/**
 * Python-C++ 간 공유 메모리 통신 클래스
 * Picamera2에서 캡처한 프레임을 C++로 전달
 */
class SharedMemory {
public:
    SharedMemory(const std::string& name, size_t size, bool create = false);
    ~SharedMemory();

    void* getBuffer();
    bool isValid() const { return fd_ != -1 && buffer_ != nullptr; }
    size_t getSize() const { return size_; }

private:
    std::string name_;
    size_t size_;
    int fd_;
    void* buffer_;
    bool owner_;

    void cleanup();
};
