cmake_minimum_required(VERSION 3.8)
project(LineFusion)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -Wextra -Wpedantic)

set(CMAKE_EXPORT_COMPILE_COMMANDS on)


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


set(SOURCE_FILE src/main.cpp)
find_package(PCL REQUIRED)
find_package (Eigen3 REQUIRED NO_MODULE)
include_directories(
    lib
    SYSTEM
    OpenMP::OpenMP_CXX
    ${PCL_INCLUDE_DIRS}
)

###########
# Library #
###########
add_library(${PROJECT_NAME}_lib
  src/LineFusion.cpp
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
target_link_libraries(${PROJECT_NAME}_lib
  ${PCL_LIBRARIES}
  Eigen3::Eigen
)

##############
# Executable #
##############

add_executable(${PROJECT_NAME} ${SOURCE_FILE} )

target_link_libraries(${PROJECT_NAME}
  ${PROJECT_NAME}_lib
)
