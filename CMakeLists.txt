# library for solving Poisson-like equations
set(LIBRARY_NAME MultiGridSolver)

# source files
set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/grid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/multigrid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/multigrid_solver.cpp
)

# create static library
add_library(${LIBRARY_NAME} STATIC ${SOURCE_FILES})

# set compile flags
target_compile_definitions(${LIBRARY_NAME} PRIVATE OPENMP)

# publish headers for this library
target_include_directories(${LIBRARY_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

# dependencies
target_link_libraries(${LIBRARY_NAME}
    ${FFTW_LIB}
)