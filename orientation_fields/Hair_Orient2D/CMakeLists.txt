cmake_minimum_required(VERSION 3.8)
project(Orient2D)

set(CMAKE_CXX_STANDARD 11)


set(SOURCE_FILES main.cpp Orient2D.hpp
        )

find_package( OpenCV REQUIRED PATHS "~/build")
        
set(OPENCV_IO_ENABLE_OPENEXR ON)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

FIND_PACKAGE(JPEG REQUIRED)
FIND_PACKAGE(PNG  REQUIRED)

INCLUDE_DIRECTORIES(${JPEG_INCLUDE_DIR}
                    ${PNG_INCLUDE_DIR}
        ${GLFW3_INCLUDE_PATH}
        ${GLEW_INCLUDE_PATH}
        ${GLM_INCLUDE_PATH}
        ${FFTW3F_INCLUDE_DIR}
        )

add_executable(Orient2D ${SOURCE_FILES} )
target_link_libraries(Orient2D
        ${OpenCV_LIBS}
        ${JPEG_LIBRARIES}
        ${PNG_LIBRARIES}
        fftw3
        )
