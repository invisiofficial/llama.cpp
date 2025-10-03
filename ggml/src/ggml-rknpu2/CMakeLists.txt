ggml_add_backend_library(ggml-rknpu2
    ggml-rknpu2.cpp
)

# Указываем CMake, где искать библиотеку librknnrt.so
link_directories(/home/orangepi/Projects/rknpu2/runtime/RK3588/Linux/librknn_api/lib/aarch64)

# Линкуемся с библиотекой
target_link_libraries(ggml-rknpu2 PRIVATE rknnrt)

# Указываем CMake, где искать заголовочные файлы (*.h)
target_include_directories(ggml-rknpu2 PRIVATE /home/orangepi/Projects/rknpu2/runtime/RK3588/Linux/librknn_api/include)