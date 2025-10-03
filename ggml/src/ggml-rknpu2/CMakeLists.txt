ggml_add_backend_library(ggml-rknpu2
    ggml-rknpu2.cpp
)

# Указываем CMake, где искать заголовочные файлы (*.h)
target_include_directories(ggml-rknpu2 PRIVATE 
    /home/orangepi/Projects/rknpu2/runtime/RK3588/Linux/librknn_api/include
)

# Линкуемся с библиотекой, указывая полный путь к ней
target_link_libraries(ggml-rknpu2 PRIVATE 
    /home/orangepi/Projects/rknpu2/runtime/RK3588/Linux/librknn_api/aarch64/librknnrt.so
)