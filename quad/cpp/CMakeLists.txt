cmake_minimum_required(VERSION 3.16)
project(quad_mpc_wbic_cpp CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# 의존성: proxddp conda env (mujoco C · eigen · eiquadprog)
set(ENV_PREFIX "/home/jsh/miniforge3/envs/proxddp" CACHE PATH "conda env prefix")

find_path(MUJOCO_INCLUDE  mujoco/mujoco.h              PATHS ${ENV_PREFIX}/include REQUIRED)
find_library(MUJOCO_LIB   mujoco                        PATHS ${ENV_PREFIX}/lib     REQUIRED)
find_path(EIGEN_INCLUDE   Eigen/Dense                   PATHS ${ENV_PREFIX}/include/eigen3 /usr/include/eigen3 REQUIRED)
find_path(EIQP_INCLUDE    eiquadprog/eiquadprog-fast.hpp PATHS ${ENV_PREFIX}/include REQUIRED)
find_library(EIQP_LIB     eiquadprog                    PATHS ${ENV_PREFIX}/lib     REQUIRED)

message(STATUS "MuJoCo: ${MUJOCO_LIB}")
message(STATUS "eiquadprog: ${EIQP_LIB}")

add_executable(quad_cpp src/main.cpp)
target_include_directories(quad_cpp PRIVATE ${MUJOCO_INCLUDE} ${EIGEN_INCLUDE} ${EIQP_INCLUDE})
# ★conda MuJoCo가 최신 libstdc++(GLIBCXX_3.4.32)로 빌드됨 → conda libstdc++ 명시 링크(시스템 g++ 11.4 것과 ABI 불일치 해결)
target_link_libraries(quad_cpp PRIVATE ${ENV_PREFIX}/lib/libstdc++.so ${MUJOCO_LIB} ${EIQP_LIB})
# 런타임에 env lib(mujoco·libstdc++) 찾도록 rpath 우선
set_target_properties(quad_cpp PROPERTIES
  BUILD_RPATH "${ENV_PREFIX}/lib"
  INSTALL_RPATH "${ENV_PREFIX}/lib")
target_link_options(quad_cpp PRIVATE "-Wl,-rpath-link,${ENV_PREFIX}/lib")
