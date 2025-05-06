file(GLOB_RECURSE CPP_SOURCES *.cpp)
add_library(${PROJECT_NAME} ${CPP_SOURCES})
target_link_libraries(${PROJECT_NAME} -lnuma)