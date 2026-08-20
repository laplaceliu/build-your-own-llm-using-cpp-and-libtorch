# reasoning/serve/CMakeLists.txt
find_package(gflags REQUIRED)

add_executable(reasoning_chat src/reasoning_chat.cpp)
target_link_libraries(reasoning_chat PRIVATE reasoning_core gflags)

add_executable(reasoning_serve src/reasoning_serve.cpp)
target_link_libraries(reasoning_serve PRIVATE reasoning_core gflags)