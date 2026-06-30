# CMakeLists.txt
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(edge_vision_surveillance)

# Specify C++ 17 Standard (Required for modern LibTorch and OpenCV)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Find the massive external C++ libraries
# (CMake will look for these in your system paths)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# 2. Tell CMake to build an executable file from your source code
add_executable(cctv_surveillance cctv_surveillance.cpp)

# 3. Mathematically link the libraries to your executable
target_link_libraries(cctv_surveillance "${TORCH_LIBRARIES}" ${OpenCV_LIBS})

# 4. Set C++ flags to enforce strict memory and compilation standards
set_property(TARGET cctv_surveillance PROPERTY CXX_STANDARD 17)

# Ensure LibTorch compiles correctly on modern systems
if (MSVC)
  file(TO_NATIVE_PATH "${TORCH_INSTALL_PREFIX}/lib" TORCH_LIB_PATH)
  target_link_directories(cctv_surveillance PRIVATE "${TORCH_LIB_PATH}")
endif()