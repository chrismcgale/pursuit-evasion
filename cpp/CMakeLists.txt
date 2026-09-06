cmake_minimum_required(VERSION 3.20)
project(pursuit_evasion_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- MuJoCo: link the exact library the Python env uses (version-matched) -----
# Defaults to the project venv; override with -DMUJOCO_ROOT=/path/to/mujoco
set(MUJOCO_ROOT "${CMAKE_SOURCE_DIR}/../.venv/lib/python3.12/site-packages/mujoco"
    CACHE PATH "Directory containing libmujoco.so.* and include/mujoco")
file(GLOB MUJOCO_LIB "${MUJOCO_ROOT}/libmujoco.so*")
if(NOT MUJOCO_LIB)
  message(FATAL_ERROR "libmujoco not found under ${MUJOCO_ROOT}. Run `uv sync` or set -DMUJOCO_ROOT.")
endif()
list(GET MUJOCO_LIB 0 MUJOCO_LIB)
message(STATUS "MuJoCo lib:      ${MUJOCO_LIB}")
message(STATUS "MuJoCo include:  ${MUJOCO_ROOT}/include")

# --- onnxruntime (system package: /usr/lib/cmake/onnxruntime) -----------------
find_package(onnxruntime QUIET)
if(NOT onnxruntime_FOUND)
  message(STATUS "onnxruntimeConfig not found; falling back to manual paths")
  set(ORT_INCLUDE /usr/include/onnxruntime CACHE PATH "onnxruntime include dir")
  set(ORT_LIB /usr/lib/libonnxruntime.so CACHE FILEPATH "libonnxruntime.so path")
endif()

# --- BehaviorTree.CPP (industry-standard BT runtime; opens in Groot2) ---------
include(FetchContent)
set(BTCPP_BUILD_TOOLS  OFF CACHE BOOL "" FORCE)
set(BTCPP_EXAMPLES     OFF CACHE BOOL "" FORCE)
set(BTCPP_UNIT_TESTS   OFF CACHE BOOL "" FORCE)
set(BTCPP_GROOT_INTERFACE ON CACHE BOOL "" FORCE)  # ZMQ publisher for Groot2 live view
FetchContent_Declare(behaviortree_cpp
  GIT_REPOSITORY https://github.com/BehaviorTree/BehaviorTree.CPP.git
  GIT_TAG 4.6.2
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(behaviortree_cpp)
# GCC 16's new -Wtemplate-body diagnostic trips on BT.CPP's bundled lexy and is
# promoted to an error by BT.CPP's -Werror. Disable -Werror for that target.
if(TARGET behaviortree_cpp)
  target_compile_options(behaviortree_cpp PRIVATE -Wno-error -Wno-template-body)
endif()

# --- core library (sim + obs + scripted + features + onnx + safety + BT) ------
add_library(pe_core
  src/sim.cpp
  src/scripted.cpp
  src/onnx_policy.cpp
  src/bt_nodes.cpp
  src/gate.cpp)
target_include_directories(pe_core PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${MUJOCO_ROOT}/include)
target_link_libraries(pe_core PUBLIC ${MUJOCO_LIB} behaviortree_cpp)
if(onnxruntime_FOUND)
  target_link_libraries(pe_core PUBLIC onnxruntime::onnxruntime)
else()
  target_include_directories(pe_core PUBLIC ${ORT_INCLUDE})
  target_link_libraries(pe_core PUBLIC ${ORT_LIB})
endif()
# find libmujoco.so.* at runtime without LD_LIBRARY_PATH gymnastics
set_target_properties(pe_core PROPERTIES BUILD_RPATH "${MUJOCO_ROOT}")

# --- runner executable --------------------------------------------------------
add_executable(pe_run src/runner.cpp)
target_link_libraries(pe_run PRIVATE pe_core)
set_target_properties(pe_run PROPERTIES BUILD_RPATH "${MUJOCO_ROOT}")

# --- unit tests (per BT node + safety filter) via doctest ---------------------
FetchContent_Declare(doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.11
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(doctest)
enable_testing()
add_executable(pe_tests tests/test_bt_nodes.cpp tests/test_safety.cpp tests/test_games.cpp)
target_link_libraries(pe_tests PRIVATE pe_core doctest::doctest)
# test_games loads the SHIPPED trees/*.xml, so a Groot2 hand-edit that breaks a
# gate profile fails the tests instead of silently changing behaviour
target_compile_definitions(pe_tests PRIVATE PE_TREES_DIR="${CMAKE_SOURCE_DIR}/trees")
set_target_properties(pe_tests PROPERTIES BUILD_RPATH "${MUJOCO_ROOT}")
add_test(NAME pe_tests COMMAND pe_tests)
