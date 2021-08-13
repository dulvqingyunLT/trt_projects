cmake_minimum_required(VERSION 3.12)
project(MaskRCNN)

set(CMAKE_CXX_STANDARD 14)
set( CMAKE_BUILD_TYPE "Debug" )

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)
set(COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../common)
include_directories(${COMMON_DIR})

# set(PLUGIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../TensorRT/build)
# include_directories(${PLUGIN_DIR})

#OpenCV
set(OpenCV_DIR /usr/share/OpenCV)
# set(OpenCV_DIR /home/ubuntu/source_code/opencv3.4/opencv/build)

find_package( OpenCV 3 REQUIRED )
include_directories(/usr/local/include/opencv2) #指定opencv3的安装目录
# include_directories(/home/ubuntu/source_code/opencv3.4/opencv/include)
set(OpenCV_LIB_PATH /usr/lib/x86_64-linux-gnu/)
# set(OpenCV_LIB_PATH /home/ubuntu/source_code/opencv3.4/opencv/build/lib)
file(GLOB OpenCV_LIBS ${OpenCV_LIB_PATH}/*.so)

# message(${OpenCV_DIR})

#set(OpenCV_DIR /home/yckj2523/lab/opencv/build)
#find_package( OpenCV REQUIRED )
##include_directories(/home/yckj2523/lab/opencv/include)
##set(OpenCV_LIB_PATH /home/yckj2523/lab/opencv/build/lib)
#file(GLOB OpenCV_LIBS ${OpenCV_LIB_PATH}/*.so)

# message(OpenCV_LIBS:${OpenCV_LIBS})


# 自己编译的tensor库，如infer_plugin
# link_directories(/home/ubuntu/source_code/trt_projects/TensorRT/build)
# set(CUSTOMIZED_TENSORRT_LIB_PATH /home/ubuntu/source_code/trt_projects/TensorRT/build)
# file(GLOB CUSTOMIZED_TENSORRT_LIBS ${CUSTOMIZED_TENSORRT_LIB_PATH}/*.so)

# TensorRT
include_directories(/usr/local/TensorRT-7.2.3.4/include)
set(TENSORRT_LIB_PATH /usr/local/TensorRT-7.2.3.4/lib)

#include_directories(/usr/local/include)
# set(TENSORRT_LIB_PATH /usr/local/lib)
file(GLOB TensorRT_LIBS ${TENSORRT_LIB_PATH}/*.so)
#message(TensorRT_LIBS:${TensorRT_LIBS})



# set(TRT_PLUGIN_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../TensorRT/build)
# file(GLOB TRT_PLUGIN_LIBS ${TRT_PLUGIN_LIB_PATH}/libnvinfer_plugin_debug.so)


# CUDA
#set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-9.0)
find_package(CUDA REQUIRED)
#message(CUDA_LIBRARIES:${CUDA_LIBRARIES})
# message(CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
#include_directories( 
#        /usr/local/cuda-9.0/include
#)
#file(GLOB LIBS /usr/local/cuda/lib64/*.so.9.0)
#message(CUDA_LIBS:${CUDA_LIBS})
link_directories(
       /usr/local/cuda-11.1/lib64
)

add_library(common_shared SHARED
       ${COMMON_DIR}/getOptions.cpp
       ${COMMON_DIR}/logger.cpp
       ${COMMON_DIR}/sampleEngines.cpp
       ${COMMON_DIR}/sampleInference.cpp
       ${COMMON_DIR}/sampleOptions.cpp
       ${COMMON_DIR}/sampleReporting.cpp)



file(GLOB EXTRA_LIBS ../build/LGTPlugin/*.so)


file(GLOB SOURCE
            main.cpp
            mask_rcnn.cpp            
            )

add_executable(${PROJECT_NAME} ${SOURCE})

target_link_libraries(${PROJECT_NAME}
       common_shared
       # ${CUSTOMIZED_TENSORRT_LIBS}
       ${TensorRT_LIBS}
       libcublas.so libcudart.so libnvrtc.so
       # libopencv_core.so libopencv_highgui.so libopencv_imgproc.so libopencv_imgcodecs.so
#        LIBS
       ${OpenCV_LIBS}
       ${EXTRA_LIBS}
       /home/yhuang/trt_projects/LGTPlugin/build/libFasterRCNNKernels.so 
       )

#include_directories(
#        /home/yckj2437/code/API/onnxruntime-linux-x64-1.3.0/include
#        ${OpenCV_INCLUDE_DIRS}
#        )
#link_directories(
#        /home/yckj2437/code/API/onnxruntime-linux-x64-1.3.0/lib
#)

