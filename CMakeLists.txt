cmake_minimum_required(VERSION 3.0.0)
project(max_diversity)

# CPLEX directories
set(CPLEX_ROOT_DIR "/opt/ibm/ILOG/CPLEX_Studio2211")
set(CPLEX_INCLUDE_DIR "${CPLEX_ROOT_DIR}/cplex/include")
set(CPLEX_LIB_DIR "${CPLEX_ROOT_DIR}/cplex/lib/x86-64_linux/static_pic")
set(CONCERT_INCLUDE_DIR "${CPLEX_ROOT_DIR}/concert/include")
set(CONCERT_LIB_DIR "${CPLEX_ROOT_DIR}/concert/lib/x86-64_linux/static_pic")

# Include directories
include_directories(${CPLEX_INCLUDE_DIR})
include_directories(${CONCERT_INCLUDE_DIR})
include_directories(${EIGEN_INCLUDE_DIR})

# Link directories
link_directories(${CPLEX_LIB_DIR})
link_directories(${CONCERT_LIB_DIR})

# Optimisation
add_compile_options(-O3)

add_executable( ct 
                src/cut_plane.cpp
                )
target_link_libraries(ct ilocplex concert cplex m pthread)

add_executable( oa 
                src/outer_approximation.cpp
                )
target_link_libraries(oa ilocplex concert cplex m pthread)

add_executable( ct_lm
                src/cut_plane_low_memory.cpp
                )
target_link_libraries(ct_lm ilocplex concert cplex m pthread)

add_executable( ct_world
                src/cut_plane_world.cpp
                )
target_link_libraries(ct_world ilocplex concert cplex m pthread)
                
add_executable( glover
                src/glover.cpp
                ) 
target_link_libraries(glover ilocplex concert cplex m pthread)

add_executable( quad
src/quad.cpp
) 
target_link_libraries(quad ilocplex concert cplex m pthread)

add_executable( obma
src/OBMA_MDP.cpp
) 
