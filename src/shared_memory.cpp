#include "shared_memory.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <iostream>

SharedMemory::SharedMemory(const std::string &name, size_t size, bool create)
    : name_("/" + name), size_(size), fd_(-1), buffer_(nullptr), owner_(create)
{

    if (create)
    {
        shm_unlink(name_.c_str());
        fd_ = shm_open(name_.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd_ == -1)
        {
            std::cerr << "공유 메모리 생성 실패: " << name_ << std::endl;
            return;
        }

        if (ftruncate(fd_, size_) == -1)
        {
            std::cerr << "공유 메모리 크기 설정 실패" << std::endl;
            close(fd_);
            fd_ = -1;
            return;
        }
    }
    else
    {
        fd_ = shm_open(name_.c_str(), O_RDONLY, 0666);
        if (fd_ == -1)
        {
            std::cerr << "공유 메모리 열기 실패: " << name_ << std::endl;
            return;
        }
    }

    int prot = create ? (PROT_READ | PROT_WRITE) : PROT_READ;
    buffer_ = mmap(nullptr, size_, prot, MAP_SHARED, fd_, 0);

    if (buffer_ == MAP_FAILED)
    {
        std::cerr << "메모리 매핑 실패" << std::endl;
        buffer_ = nullptr;
        close(fd_);
        fd_ = -1;
        return;
    }

    std::cout << "공유 메모리 초기화: " << name_ << " (" << size_ << " bytes)" << std::endl;
}

SharedMemory::~SharedMemory()
{
    cleanup();
}

void SharedMemory::cleanup()
{
    if (buffer_ != nullptr && buffer_ != MAP_FAILED)
    {
        munmap(buffer_, size_);
        buffer_ = nullptr;
    }

    if (fd_ != -1)
    {
        close(fd_);
        fd_ = -1;
    }

    if (owner_)
        shm_unlink(name_.c_str());
}

void *SharedMemory::getBuffer()
{
    return buffer_;
}