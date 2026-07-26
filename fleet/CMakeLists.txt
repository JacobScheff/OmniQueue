cmake_minimum_required(VERSION 3.15)
project(fleet_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_fleet_sim
    native/src/bindings.cpp
)

# Headers live at the fleet/ root (City.hpp, Simulation.hpp, FleetEnv.hpp, ...).
target_include_directories(_fleet_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(MSVC)
    target_compile_options(_fleet_sim PRIVATE /O2 /EHsc /bigobj)
else()
    target_compile_options(_fleet_sim PRIVATE -O3 -Wall -Wextra)
endif()

install(TARGETS _fleet_sim LIBRARY DESTINATION . RUNTIME DESTINATION . ARCHIVE DESTINATION .)
