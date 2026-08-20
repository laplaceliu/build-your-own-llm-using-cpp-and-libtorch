# reasoning/eval/CMakeLists.txt
find_package(gflags REQUIRED)

add_executable(eval_math src/eval_math.cpp)
target_link_libraries(eval_math PRIVATE reasoning_core gflags)