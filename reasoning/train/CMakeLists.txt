# reasoning/train/CMakeLists.txt
find_package(gflags REQUIRED)

add_executable(sft_train src/sft_train.cpp)
target_link_libraries(sft_train PRIVATE reasoning_core gflags)

add_executable(rl_train src/rl_train.cpp)
target_link_libraries(rl_train PRIVATE reasoning_core gflags)