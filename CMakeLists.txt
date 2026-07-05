cmake_minimum_required(VERSION 3.15)
project(omniqueue_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_park_sim
    native/src/bindings.cpp
    native/src/park_sim.cpp
)

target_include_directories(_park_sim PRIVATE native/include native/generated)

if(MSVC)
    target_compile_options(_park_sim PRIVATE /O2 /EHsc)
else()
    target_compile_options(_park_sim PRIVATE -O3 -Wall -Wextra)
endif()

install(TARGETS _park_sim LIBRARY DESTINATION . RUNTIME DESTINATION . ARCHIVE DESTINATION .)
