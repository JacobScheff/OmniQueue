cmake_minimum_required(VERSION 3.15)
project(omniqueue_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

if(DEFINED ENV{TORCH_CMAKE_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{TORCH_CMAKE_PREFIX}")
else()
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TORCH_PREFIX_RESULT
        ERROR_VARIABLE TORCH_PREFIX_ERROR
    )
    if(NOT TORCH_PREFIX_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Failed to locate PyTorch CMake prefix. Install torch, then either:\n"
            "  export TORCH_CMAKE_PREFIX=$(python -c \"import torch; print(torch.utils.cmake_prefix_path)\")\n"
            "  pip install -e . --no-build-isolation\n"
            "Error: ${TORCH_PREFIX_ERROR}")
    endif()
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
endif()

set(TORCH_FIND_CUDA OFF CACHE BOOL "Link CPU LibTorch when CUDA toolkit is unavailable")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
    OUTPUT_VARIABLE TORCH_INSTALL_LIB
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

pybind11_add_module(_park_sim
    native/src/bindings.cpp
    native/src/park_sim.cpp
    native/src/policy_runner.cpp
)

target_include_directories(_park_sim PRIVATE native/include native/generated)
target_compile_definitions(_park_sim PRIVATE OMNIQUEUE_LIBTORCH=1)
target_link_libraries(_park_sim PRIVATE ${TORCH_LIBRARIES})
set_property(TARGET _park_sim PROPERTY CXX_STANDARD 17)
set_target_properties(_park_sim PROPERTIES
    BUILD_RPATH "${TORCH_INSTALL_LIB}"
    INSTALL_RPATH "${TORCH_INSTALL_LIB}"
)

if(MSVC)
    target_compile_options(_park_sim PRIVATE /O2 /EHsc)
else()
    target_compile_options(_park_sim PRIVATE -O3 -Wall -Wextra)
endif()

install(TARGETS _park_sim LIBRARY DESTINATION . RUNTIME DESTINATION . ARCHIVE DESTINATION .)
