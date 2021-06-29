
# file(GLOB SRCS *.cpp)
# set(PLUGIN_SOURCES ${PLUGIN_SOURCES} ${SRCS})
# set(PLUGIN_SOURCES ${PLUGIN_SOURCES} PARENT_SCOPE)



cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 14)
set( CMAKE_BUILD_TYPE "Debug" )

project(FasterRCNNKernels)

find_package( OpenCV 3 REQUIRED )
include_directories(/usr/local/include/opencv) #指定opencv3的安装目录
# include_directories(/home/ubuntu/source_code/opencv3.4/opencv/include)
set(OpenCV_LIB_PATH /usr/local/lib)
# set(OpenCV_LIB_PATH /home/ubuntu/source_code/opencv3.4/opencv/build/lib)
file(GLOB OpenCV_LIBS ${OpenCV_LIB_PATH}/*.so)


# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
#include_directories(
#        /usr/local/cuda-9.0/include
#)
link_directories(
       /usr/local/cuda-11.0/lib64
)

set(CUB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../cub-1.8.0/cub)
include_directories(${CUB_DIR})
# include_directories(cub)
#message(CUB_ROOT_DIR:${CUB_ROOT_DIR})
#add_subdirectory(cub cub.out)

include_directories(/home/ubuntu/source_code/TensorRT-7.2.2.3/include)
set(TENSORRT_LIB_PATH /home/ubuntu/source_code/TensorRT-7.2.2.3/lib)
#include_directories(/usr/local/include)
# set(TENSORRT_LIB_PATH /usr/local/lib)
file(GLOB TensorRT_LIBS ${TENSORRT_LIB_PATH}/*.so)

file(GLOB_RECURSE CURRENT_HEADERS *.cuh *.h)
file(GLOB_RECURSE CURRENT_SOURCES *.cu *.cpp)

list(APPEND CUDA_NVCC_FLAGS "-std=c++14")
#set(CUDA_NVCC_FLAGS;-G;-g;-std=c++11)
source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

cuda_add_library(${PROJECT_NAME} SHARED  ${CURRENT_SOURCES} ${TensorRT_LIBS})

# install(TARGETS ${PROJECT_NAME})
include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME}  
       # ARCHIVE DESTINATION  ${CMAKE_INSTALL_PREFIX}
       LIBRARY DESTINATION /home/ubuntu
       # RUNTIME DESTINATION /home/ubuntu
        )

message(hahha)