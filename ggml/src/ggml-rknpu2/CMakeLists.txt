ggml_add_backend_library(ggml-rknpu2
    ggml-rknpu2.cpp
)

# Укажите путь к библиотеке rknnrt, если она не в системном пути
# find_library(RKNNRT_LIBRARY rknnrt HINTS /path/to/your/rknpu2/lib)
# target_link_libraries(ggml-rknpu2 PRIVATE ${RKNNRT_LIBRARY})
target_link_libraries(ggml-rknpu2 PRIVATE rknnrt)

# Укажите путь к заголовочным файлам, если они не в системном пути
# target_include_directories(ggml-rknpu2 PRIVATE /path/to/your/rknpu2/include)