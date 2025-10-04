cmake_minimum_required(VERSION 3.16)
project(MediaIntegrity VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})

add_executable(camera_test 
    test/camera_test.cpp 
    src/shared_memory.cpp
)

target_link_libraries(camera_test
    ${OpenCV_LIBS}
    Threads::Threads
    rt
)