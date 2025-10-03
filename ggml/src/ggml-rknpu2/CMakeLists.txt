ggml_add_backend_library(ggml-rknpu2
    ggml-rknpu2.cpp
)

# Явно ищем библиотеку в указанной директории
find_library(RKNPU_LIBRARY
    NAMES rknnrt
    HINTS /home/orangepi/rknpu2/runtime/RK3588/Linux/librknn_api/lib/aarch64
)

if (RKNPU_LIBRARY)
    message(STATUS "Found RKNPU library: ${RKNPU_LIBRARY}")
    # Линкуемся с библиотекой, используя полный путь
    target_link_libraries(ggml-rknpu2 PRIVATE ${RKNPU_LIBRARY})
else()
    message(FATAL_ERROR "RKNPU library (rknnrt) not found. Please check the path.")
endif()

# Указываем CMake, где искать заголовочные файлы (*.h)
target_include_directories(ggml-rknpu2 PRIVATE /home/orangepi/rknpu2/runtime/RK3588/Linux/librknn_api/include)