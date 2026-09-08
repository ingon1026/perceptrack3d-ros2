# PercepTrack3D Phase 10 — C++ 투영/ROI/frustum 컴포넌트.
# 빌드: scripts/build_cpp.sh  (= cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
#        -DPython_EXECUTABLE=.venv/bin/python -Dpybind11_DIR=$(.venv/bin/python -m pybind11 --cmakedir) && cmake --build cpp/build -j)
cmake_minimum_required(VERSION 3.18)
project(pt3d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# 벤치마크 재현성을 위해 Release 플래그를 고정한다. -march=native 는 쓰지 않는다 (x86-64 기본 SSE2, 다른 PC 에서도 같은 코드).
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# (1) 핵심 정적 라이브러리 — Python 에 의존하지 않는다. ROS2 C++ 노드에서도 그대로 링크할 수 있다.
add_library(pt3d_core STATIC src/projection.cpp)
target_include_directories(pt3d_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(pt3d_core PUBLIC Eigen3::Eigen)
target_compile_options(pt3d_core PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(pt3d_core PROPERTIES POSITION_INDEPENDENT_CODE ON)  # 파이썬 .so 에 링크되므로 PIC 필요

# (2) 단위 테스트 (GoogleTest: sudo apt install libgtest-dev)
enable_testing()
find_package(GTest REQUIRED)
add_executable(pt3d_tests tests/test_projection.cpp)
target_link_libraries(pt3d_tests PRIVATE pt3d_core GTest::gtest_main)
target_compile_options(pt3d_tests PRIVATE -Wall -Wextra)
add_test(NAME pt3d_tests COMMAND pt3d_tests)

# (3) pybind11 확장 모듈 pt3d_cpp → cpp/build/pt3d_cpp.cpython-*.so
#     Python 은 -DPython_EXECUTABLE 로 지정한 venv 인터프리터를 쓴다 (셸 PYTHONPATH 의 ROS2 python 을 잡지 않도록).
set(PYBIND11_FINDPYTHON ON)
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(pt3d_cpp src/bindings.cpp)
target_link_libraries(pt3d_cpp PRIVATE pt3d_core)
target_compile_options(pt3d_cpp PRIVATE -Wall -Wextra)
