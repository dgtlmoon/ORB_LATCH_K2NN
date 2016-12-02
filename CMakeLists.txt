cmake_minimum_required(VERSION 3.6)
project(ORB_LATCH_K2NN)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wshadow -pedantic -Ofast -std=gnu++17 -fomit-frame-pointer -mavx2 -march=native -mfma -flto -funroll-all-loops -fpeel-loops -ftracer -ftree-vectorize")

set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} m pthread)

set(SOURCE_FILES main.cpp)
add_executable(ORB_LATCH_K2NN ${SOURCE_FILES})

TARGET_LINK_LIBRARIES(ORB_LATCH_K2NN pthread opencv_core  opencv_highgui opencv_imgcodecs boost_system boost_filesystem opencv_features2d)
