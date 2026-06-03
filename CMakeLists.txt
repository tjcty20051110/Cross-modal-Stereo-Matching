cmake_minimum_required(VERSION 3.18)

project(cross-modal-stereo-matching
    VERSION 0.1.0
    DESCRIPTION "Cross-Modal (RGB + IR) Stereo Matching System"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

option(CMS_BUILD_TESTS           "Build unit and property tests"        ON)
option(CMS_ENABLE_ONNXRUNTIME    "Enable ONNX Runtime DL backend"       OFF)
option(CMS_ENABLE_LIBTORCH       "Enable LibTorch DL backend"           OFF)
option(CMS_ENABLE_WARNINGS       "Enable extra compiler warnings"       ON)
option(CMS_WARNINGS_AS_ERRORS    "Treat compiler warnings as errors"    OFF)
option(CMS_ENABLE_PBT_TESTS      "Enable property-based tests via rapidcheck" OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(CMS_ENABLE_WARNINGS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor)
        if(CMS_WARNINGS_AS_ERRORS)
            add_compile_options(-Werror)
        endif()
    elseif(MSVC)
        add_compile_options(/W4)
        if(CMS_WARNINGS_AS_ERRORS)
            add_compile_options(/WX)
        endif()
    endif()
endif()

find_package(OpenCV 4.0 REQUIRED COMPONENTS core imgproc imgcodecs highgui calib3d dnn)
message(STATUS "Found OpenCV ${OpenCV_VERSION} at ${OpenCV_DIR}")

find_package(yaml-cpp CONFIG QUIET)
if(NOT yaml-cpp_FOUND)
    find_package(yaml-cpp REQUIRED)
endif()
if(NOT TARGET yaml-cpp::yaml-cpp AND TARGET yaml-cpp)
    add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
endif()
message(STATUS "Found yaml-cpp")

if(CMS_ENABLE_ONNXRUNTIME)
    find_package(onnxruntime CONFIG QUIET)
    if(NOT onnxruntime_FOUND)
        message(WARNING
            "CMS_ENABLE_ONNXRUNTIME=ON but ONNX Runtime was not found via find_package(onnxruntime CONFIG). "
            "Set onnxruntime_DIR to the directory containing onnxruntimeConfig.cmake, or disable with -DCMS_ENABLE_ONNXRUNTIME=OFF.")
    else()
        message(STATUS "Found ONNX Runtime")
        add_compile_definitions(CMS_HAS_ONNXRUNTIME=1)
    endif()
endif()

if(CMS_ENABLE_LIBTORCH)
    find_package(Torch QUIET)
    if(NOT Torch_FOUND)
        message(WARNING
            "CMS_ENABLE_LIBTORCH=ON but LibTorch was not found via find_package(Torch). "
            "Set Torch_DIR to the directory containing TorchConfig.cmake, or disable with -DCMS_ENABLE_LIBTORCH=OFF.")
    else()
        message(STATUS "Found LibTorch ${TORCH_VERSION}")
        add_compile_definitions(CMS_HAS_LIBTORCH=1)
    endif()
endif()

add_subdirectory(src)

if(CMS_BUILD_TESTS OR CMS_ENABLE_PBT_TESTS)
    enable_testing()
    include(CTest)
    add_subdirectory(tests)
endif()

install(DIRECTORY configs/
    DESTINATION share/${PROJECT_NAME}/configs
    FILES_MATCHING PATTERN "*.yaml"
)

message(STATUS "")
message(STATUS "=== ${PROJECT_NAME} v${PROJECT_VERSION} ===")
message(STATUS "  Build type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:        ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build tests:         ${CMS_BUILD_TESTS}")
message(STATUS "  PBT tests:           ${CMS_ENABLE_PBT_TESTS}")
message(STATUS "  ONNX Runtime:        ${CMS_ENABLE_ONNXRUNTIME}")
message(STATUS "  LibTorch:            ${CMS_ENABLE_LIBTORCH}")
message(STATUS "  Install prefix:      ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
