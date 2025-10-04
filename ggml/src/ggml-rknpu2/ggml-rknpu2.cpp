#include "ggml-backend-impl.h"
#include "ggml-rknpu2.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#include <cassert>
#include <cstring>
#include <vector>
#include <memory>
#include <stdexcept>

// RKNPU2 specific headers
#include <rknn_api.h>
#include <rknn_matmul_api.h>

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <arm_neon.h>

#define GGML_RKNPU2_INPUT_SCALE 1.7f

//================================================================================
// DMA & RKNPU Kernel Management
//================================================================================

// DMA buffer allocation logic
struct dma_heap_allocation_data {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};

#define DMA_HEAP_IOC_MAGIC      'H'
#define DMA_HEAP_IOCTL_ALLOC    _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct dma_heap_allocation_data)
#define DMA_BUF_SYNC_READ      (1 << 0)
#define DMA_BUF_SYNC_WRITE     (2 << 0)
#define DMA_BUF_SYNC_RW        (DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE)
#define DMA_BUF_SYNC_START     (0 << 2)
#define DMA_BUF_SYNC_END       (1 << 2)
#define DMA_BUF_BASE           'b'
#define DMA_BUF_IOCTL_SYNC     _IOW(DMA_BUF_BASE, 0, uint64_t)

static int dma_alloc(size_t size, int *fd, void **va) {
    int ret;
    int prot;
    void *mmap_va;
    int dma_heap_fd = -1;
    struct dma_heap_allocation_data buf_data;
    const char* path = "/dev/dma_heap/system";

    dma_heap_fd = open(path, O_RDWR);
    if (dma_heap_fd < 0) {
        fprintf(stderr, "open %s fail!\n", path);
        return dma_heap_fd;
    }

    memset(&buf_data, 0x0, sizeof(struct dma_heap_allocation_data));
    buf_data.len = size;
    buf_data.fd_flags = O_CLOEXEC | O_RDWR;
    ret = ioctl(dma_heap_fd, DMA_HEAP_IOCTL_ALLOC, &buf_data);
    if (ret < 0) {
        fprintf(stderr, "RK_DMA_HEAP_ALLOC_BUFFER failed\n");
        close(dma_heap_fd);
        return ret;
    }

    prot = PROT_READ | PROT_WRITE;
    mmap_va = (void *)mmap(NULL, buf_data.len, prot, MAP_SHARED, buf_data.fd, 0);
    if (mmap_va == MAP_FAILED) {
        fprintf(stderr, "mmap failed: %s\n", strerror(errno));
        close(dma_heap_fd);
        return -errno;
    }

    *va = mmap_va;
    *fd = buf_data.fd;
    close(dma_heap_fd);
    return 0;
}

static void dma_buf_free(size_t size, int *fd, void *va) {
    if (*fd >= 0) {
        munmap(va, size);
        close(*fd);
        *fd = -1;
    }
}

// MatMul kernel management
struct ggml_rknpu2_matmul_kernel {
    rknn_matmul_info matmul_info;
    rknn_matmul_ctx matmul_ctx;
    rknn_matmul_io_attr matmul_io_attr;

    rknn_tensor_mem* A;
    rknn_tensor_mem* C;
};

#define GGML_RKNPU2_MAX_MATMUL_KERNELS 16
static struct ggml_rknpu2_matmul_kernel matmul_kernels[GGML_RKNPU2_MAX_MATMUL_KERNELS];
static int matmul_kernels_count = 0;

static struct ggml_rknpu2_matmul_kernel* ggml_rknpu2_matmul_kernel_find(int m, int k, int n, rknn_matmul_type type) {
    for (int i = 0; i < matmul_kernels_count; i++) {
        struct ggml_rknpu2_matmul_kernel* kernel = &matmul_kernels[i];
        if (kernel->matmul_info.M == m && kernel->matmul_info.K == k && kernel->matmul_info.N == n &&
            type == kernel->matmul_info.type)
            return kernel;
    }
    return NULL;
}

static struct ggml_rknpu2_matmul_kernel* ggml_rknpu2_matmul_kernel_create(int m, int k, int n, rknn_matmul_type type) {
    struct ggml_rknpu2_matmul_kernel* kernel = ggml_rknpu2_matmul_kernel_find(m, k, n, type);
    if (kernel != NULL) return kernel;

    GGML_ASSERT(matmul_kernels_count < GGML_RKNPU2_MAX_MATMUL_KERNELS);
    kernel = &matmul_kernels[matmul_kernels_count++];
    memset(kernel, 0, sizeof(struct ggml_rknpu2_matmul_kernel));

    kernel->matmul_info.M = m;
    kernel->matmul_info.K = k;
    kernel->matmul_info.N = n;
    kernel->matmul_info.type = type;
    kernel->matmul_info.B_layout = RKNN_MM_LAYOUT_NATIVE;
    kernel->matmul_info.AC_layout = RKNN_MM_LAYOUT_NORM;

    int ret = rknn_matmul_create(&kernel->matmul_ctx, &kernel->matmul_info, &kernel->matmul_io_attr);
    GGML_ASSERT(ret == 0);
    // TODO: Make core mask configurable
    rknn_matmul_set_core_mask(kernel->matmul_ctx, RKNN_NPU_CORE_ALL);
    printf("Created RKNPU2 matmul kernel: M=%d, K=%d, N=%d\n", m, k, n);

    kernel->A = rknn_create_mem(kernel->matmul_ctx, kernel->matmul_io_attr.A.size);
    kernel->C = rknn_create_mem(kernel->matmul_ctx, kernel->matmul_io_attr.C.size);

    ret = rknn_matmul_set_io_mem(kernel->matmul_ctx, kernel->A, &kernel->matmul_io_attr.A);
    GGML_ASSERT(ret == 0);
    ret = rknn_matmul_set_io_mem(kernel->matmul_ctx, kernel->C, &kernel->matmul_io_attr.C);
    GGML_ASSERT(ret == 0);
    return kernel;
}

static void ggml_rknpu2_destroy() {
    for (size_t i = 0; i < matmul_kernels_count; i++) {
        struct ggml_rknpu2_matmul_kernel* kernel = &matmul_kernels[i];
        rknn_destroy_mem(kernel->matmul_ctx, kernel->A);
        rknn_destroy_mem(kernel->matmul_ctx, kernel->C);
        rknn_matmul_destroy(kernel->matmul_ctx);
    }
    matmul_kernels_count = 0;
}

// Data transformation logic
static void ggml_rknpu2_reorder_q8_0_to_native_int8(
    int8_t * dst,
    const void * src_q8_0,
    size_t k, size_t n,
    float d_max) {

    GGML_ASSERT(k > 0 && n > 0 && k % 32 == 0 && n % 32 == 0);
    GGML_ASSERT(d_max > 0.0f);

    const size_t rknpu_strides[4] = {k / 32 * 32 * 32, 32 * 32, 32, 1};
    const auto q8_0_traits = ggml_get_type_traits(GGML_TYPE_Q8_0);

    const size_t nb_per_row = k / 32;
    const float scale_factor = 127.0f / d_max;

    // The source tensor is laid out as [N, K]
    for (size_t row = 0; row < n; ++row) {
        for (size_t col_block = 0; col_block < nb_per_row; ++col_block) {
            const size_t k_base = col_block * 32;
            const block_q8_0 * block = (const block_q8_0 *)((const char *)src_q8_0 + (row * nb_per_row + col_block) * q8_0_traits->blck_size);

            const float d_local = ggml_fp16_to_fp32(block->d);

            for (size_t i = 0; i < 32; ++i) {
                const float val_f32 = block->qs[i] * d_local;
                const int8_t val_int8 = (int8_t)roundf(fminf(fmaxf(val_f32 * scale_factor, -127.0f), 127.0f));

                // Calculate destination index based on NPU native layout [N/32, K/32, 32, 32]
                const size_t current_k = k_base + i;
                const size_t current_n = row;

                const size_t n_chunk = current_n / 32;
                const size_t k_chunk = current_k / 32;
                const size_t n_inner = current_n % 32;
                const size_t k_inner = current_k % 32;

                const size_t dst_idx = n_chunk * rknpu_strides[0] +
                                       k_chunk * rknpu_strides[1] +
                                       n_inner * rknpu_strides[2] +
                                       k_inner * rknpu_strides[3];

                dst[dst_idx] = val_int8;
            }
        }
    }
}

//================================================================================
// GGML Backend Implementation
//================================================================================

// Backend context
struct ggml_backend_rknpu2_context {
    int core_mask = RKNN_NPU_CORE_ALL;
};

// Tensor extra data (stores prepared weights)
struct ggml_backend_rknpu2_tensor_extra {
    rknn_tensor_mem* b_mem;
    bool initialized;
    float d_max;
};

// Buffer context (stores info about a DMA allocation)
struct ggml_backend_rknpu2_buffer_context {
    int fd = -1;
    void * va = nullptr;
    size_t size = 0;
    std::vector<ggml_backend_rknpu2_tensor_extra*> created_extras;
};

//================================================================================
// Реализация интерфейсов
//================================================================================

// --- Buffer Interface ---

static void ggml_backend_rknpu2_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
    memset((char *)tensor->data + offset, value, size);
}

static void ggml_backend_rknpu2_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_backend_rknpu2_buffer_context *)buffer->context;
    memset(ctx->va, value, ctx->size);
}

static void ggml_backend_rknpu2_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_rknpu2_buffer_context *)buffer->context;

    if (!ctx->created_extras.empty()) {
        rknn_matmul_ctx temp_ctx;
        rknn_matmul_info temp_info = {};
        temp_info.type = RKNN_INT8_MM_INT8_TO_INT32;
        if (rknn_matmul_create(&temp_ctx, &temp_info, nullptr) == 0) {
            for (auto* extra : ctx->created_extras) {
                if (extra && extra->b_mem) {
                    rknn_destroy_mem(temp_ctx, extra->b_mem);
                }
                delete extra;
            }
            rknn_matmul_destroy(temp_ctx);
        }
    }

    dma_buf_free(ctx->size, &ctx->fd, ctx->va);
    delete ctx;
}

static void * ggml_backend_rknpu2_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_rknpu2_buffer_context *)buffer->context;
    return ctx->va;
}

static void ggml_backend_rknpu2_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    // Эта функция копирует данные с CPU (из `data`) в наш DMA-буфер.
    // `tensor->data` уже указывает на правильное место внутри DMA-буфера.
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
    memcpy((char *)tensor->data + offset, data, size);
}

static void ggml_backend_rknpu2_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
    memcpy(data, (const char *)tensor->data + offset, size);
}

// Эта функция вызывается один раз при размещении тензора в буфере.
// Здесь мы преобразуем веса в нативный формат NPU.
static ggml_status ggml_backend_rknpu2_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    // Мы обрабатываем только тензоры весов для mul_mat
    if ((tensor->flags & GGML_TENSOR_FLAG_PARAM) == 0) {
        return GGML_STATUS_SUCCESS; // Обрабатываем только тензоры с флагом PARAM
    }
    
    if (tensor->type != GGML_TYPE_Q8_0 || ggml_n_dims(tensor) != 2) {
        return GGML_STATUS_SUCCESS; // Поддерживаем только 2D Q8_0 тензоры
    }
    
    const int64_t k = tensor->ne[1];
    const int64_t n = tensor->ne[0];

    if (k % 32 != 0 || n % 32 != 0) {
        return GGML_STATUS_SUCCESS; // Неподходящий тип или размер
    }
    
    // 1. Деквантизация в F32
    size_t nelements = ggml_nelements(tensor);
    std::vector<float> fdata(nelements);
    ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(tensor->data, fdata.data(), nelements);

    // 2. Трансформация в нативный INT8 формат RKNPU
    std::vector<int8_t> reordered_data(nelements);
    //ggml_rknpu2_transposed_to_native_int8(reordered_data.data(), fdata.data(), k, n);

    // 3. Создаем ядро, чтобы получить matmul_ctx и размер B
    //struct ggml_rknpu2_matmul_kernel* kernel = ggml_rknpu2_matmul_kernel_create(1, k, n, RKNN_TENSOR_INT8);

    // 4. Создаем rknn_tensor_mem из нашего DMA-буфера
    //auto * buffer_ctx = (ggml_backend_rknpu2_buffer_context *)buffer->context;
    //size_t tensor_offset = (uintptr_t)tensor->data - (uintptr_t)buffer_ctx->va;

    //rknn_tensor_mem* b_mem = rknn_create_mem_from_fd(kernel->matmul_ctx, buffer_ctx->fd, buffer_ctx->va,
    //                                             kernel->matmul_io_attr.B.size, tensor_offset);

    // 5. Копируем преобразованные данные в DMA-память
    //memcpy(b_mem->virt_addr, reordered_data.data(), kernel->matmul_io_attr.B.size);

    // 6. Сохраняем указатель на подготовленные веса в extra
    //auto * extra = new ggml_backend_rknpu2_tensor_extra{b_mem};
    //tensor->extra = extra;
    //buffer_ctx->created_extras.push_back(extra);
    
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rknpu2_buffer_reset(ggml_backend_buffer_t buffer) {
    // Эта функция должна очищать все 'extra' данные, связанные с тензорами в этом буфере
    // TODO: ggml пока не предоставляет простого способа итерировать по тензорам в буфере.
    // На данный момент, утечка 'extra' данных не критична, так как буферы живут столько же, сколько и модель.
}

static const struct ggml_backend_buffer_i rknpu2_buffer_interface = {
    /* .free_buffer   = */ ggml_backend_rknpu2_buffer_free_buffer,
    /* .get_base       = */ ggml_backend_rknpu2_buffer_get_base,
    /* .init_tensor    = */ nullptr,
    /* .memset_tensor  = */ ggml_backend_rknpu2_buffer_memset_tensor, // Используем реализацию по умолчанию
    /* .set_tensor     = */ ggml_backend_rknpu2_buffer_set_tensor, // Используем реализацию по умолчанию
    /* .get_tensor     = */ ggml_backend_rknpu2_buffer_get_tensor, // Используем реализацию по умолчанию
    /* .cpy_tensor     = */ nullptr, // Используем реализацию по умолчанию
    /* .clear          = */ ggml_backend_rknpu2_buffer_clear, // Используем реализацию по умолчанию
    /* .reset          = */ ggml_backend_rknpu2_buffer_reset,
};

// --- Buffer Type Interface ---

static const char * ggml_backend_rknpu2_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "RKNPU2";
}

static ggml_backend_buffer_t ggml_backend_rknpu2_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * ctx = new ggml_backend_rknpu2_buffer_context();
    if (dma_alloc(size, &ctx->fd, &ctx->va) != 0) {
        delete ctx;
        GGML_LOG_ERROR("%s: failed to allocate DMA buffer of size %zu\n", __func__, size);
        return nullptr;
    }
    ctx->size = size;

    return ggml_backend_buffer_init(buft, rknpu2_buffer_interface, ctx, size);
}

static size_t ggml_backend_rknpu2_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    // RKNPU2 требует выравнивания, но dma_alloc справляется с этим.
    // Для ggml можно вернуть безопасное значение.
    return 64;
}

static const struct ggml_backend_buffer_type_i rknpu2_buffer_type_interface = {
    /* .get_name       = */ ggml_backend_rknpu2_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_rknpu2_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_rknpu2_buffer_type_get_alignment,
    /* .get_max_size   = */ nullptr, // defaults to SIZE_MAX
    /* .get_alloc_size = */ nullptr, // defaults to ggml_nbytes
    /* .is_host        = */ nullptr, // defaults to false
};

// --- Backend Interface ---

static const char * ggml_backend_rknpu2_get_name(ggml_backend_t backend) {
    return "RKNPU2";
}

static void ggml_backend_rknpu2_free(ggml_backend_t backend) {
    ggml_rknpu2_destroy();
    delete (ggml_backend_rknpu2_context *)backend->context;
    delete backend;
}

static ggml_status ggml_backend_rknpu2_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_MUL_MAT) {
            struct ggml_tensor * src0 = node->src[0]; // веса
            struct ggml_tensor * src1 = node->src[1]; // активации
            struct ggml_tensor * dst = node;

            // --- ЛЕНИВАЯ ИНИЦИАЛИЗАЦИЯ ВЕСОВ (src0) ---
            if (src0->extra == nullptr) {
                fprintf(stderr, "[RKNPU2] Lazily initializing weight tensor '%s'\n", src0->name);

                // Проверяем, подходит ли тензор
                const int64_t k = src0->ne[1];
                const int64_t n = src0->ne[0];

                if (src0->type != GGML_TYPE_Q8_0 || ggml_n_dims(src0) != 2 || k % 32 != 0 || n % 32 != 0) {
                     GGML_LOG_ERROR("%s: unsupported weight tensor '%s' for RKNPU2\n", __func__, src0->name);
                     return GGML_STATUS_FAILED;
                }

                // 1. НАХОДИМ ГЛОБАЛЬНЫЙ МАСШТАБ (d_max)
                float d_max = 0.0f;
                const size_t nelements = ggml_nelements(src0);
                const size_t n_blocks = nelements / 32;
                const block_q8_0 * blocks = (const block_q8_0 *)src0->data;

                for (size_t j = 0; j < n_blocks; ++j) {
                    float d_local = fabsf(ggml_fp16_to_fp32(blocks[j].d));
                    if (d_local > d_max) {
                        d_max = d_local;
                    }
                }
                GGML_ASSERT(d_max > 0.0f);

                // 2. Создаем ядро, чтобы получить matmul_ctx и размер B
                struct ggml_rknpu2_matmul_kernel* kernel = ggml_rknpu2_matmul_kernel_create(1, k, n, RKNN_INT8_MM_INT8_TO_INT32);

                // 3. Выделяем DMA-память для преобразованных весов
                rknn_tensor_mem* b_mem = rknn_create_mem(kernel->matmul_ctx, kernel->matmul_io_attr.B.size);
                GGML_ASSERT(b_mem != nullptr);

                // 4. Трансформируем веса с новым методом
                ggml_rknpu2_reorder_q8_0_to_native_int8((int8_t*)b_mem->virt_addr, src0->data, k, n, d_max);

                // 5. Создаем и сохраняем extra
                auto * extra = new ggml_backend_rknpu2_tensor_extra{b_mem, true, d_max};
                src0->extra = extra;
            }

            GGML_ASSERT(src0->extra != nullptr && "RKNPU2: weight tensor not prepared");
            auto * tensor_extra = (ggml_backend_rknpu2_tensor_extra*) src0->extra;

            const int64_t m = src1->ne[1];
            const int64_t k = src0->ne[1];
            const int64_t n = src0->ne[0];

            struct ggml_rknpu2_matmul_kernel* kernel = ggml_rknpu2_matmul_kernel_create(m, k, n, RKNN_INT8_MM_INT8_TO_INT32);
            GGML_ASSERT(kernel != nullptr);

            // Подготовка активаций (матрица A): F32 -> INT8 с динамическим масштабом
            const float * src1_data = (const float *) src1->data;
            int8_t * a_virt = (int8_t *) kernel->A->virt_addr;
            const size_t ne1 = m * k;

            // 1. Находим максимальное абсолютное значение в активациях (amax)
            float amax = 0.0f;
            for (size_t j = 0; j < ne1; j++) {
                amax = fmaxf(amax, fabsf(src1_data[j]));
            }

            // 2. Вычисляем динамический масштаб и квантизируем
            const float scale_act = amax / 127.0f;
            const float iscale_act = 1.0f / (scale_act + 1e-6f);

            for (size_t j = 0; j < ne1; j++) {
                float val_quant = src1_data[j] * iscale_act;
                a_virt[j] = (int8_t)roundf(fmaxf(-127.0f, fminf(127.0f, val_quant)));
            }

            // Установка IO-памяти
            int ret;
            ret = rknn_matmul_set_io_mem(kernel->matmul_ctx, kernel->A, &kernel->matmul_io_attr.A); GGML_ASSERT(ret == 0);
            ret = rknn_matmul_set_io_mem(kernel->matmul_ctx, tensor_extra->b_mem, &kernel->matmul_io_attr.B); GGML_ASSERT(ret == 0);
            ret = rknn_matmul_set_io_mem(kernel->matmul_ctx, kernel->C, &kernel->matmul_io_attr.C); GGML_ASSERT(ret == 0);

            // Запуск
            ret = rknn_matmul_run(kernel->matmul_ctx); GGML_ASSERT(ret == 0);

            // Обработка результата (матрица C): INT32 -> F32
            float * dst_data = (float *) dst->data;
            int32_t * c_virt = (int32_t *) kernel->C->virt_addr;
            const size_t ne_dst = m * n;

            const float scale_c = (scale_act * tensor_extra->d_max) / (127.0f * 127.0f);

            for (size_t j = 0; j < ne_dst; j++) {
                dst_data[j] = (float)c_virt[j] * scale_c;
            }
        }
    }
    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i rknpu2_backend_interface = {
    /* .get_name           = */ ggml_backend_rknpu2_get_name,
    /* .free               = */ ggml_backend_rknpu2_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
    /* .synchronize        = */ nullptr,
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ ggml_backend_rknpu2_graph_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
    /* .graph_optimize     = */ nullptr,
};

// --- Device Interface ---

static const char * ggml_backend_rknpu2_device_get_name(ggml_backend_dev_t dev) {
    return "RKNPU2";
}

static const char * ggml_backend_rknpu2_device_get_description(ggml_backend_dev_t dev) {
    return "Rockchip NPU";
}

static enum ggml_backend_dev_type ggml_backend_rknpu2_device_get_type(ggml_backend_dev_t dev) {
    // GGML_BACKEND_DEVICE_TYPE_ACCEL - это общий тип для ускорителей
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

// --- GUID ---
static ggml_guid_t ggml_backend_rknpu2_guid(void) {
    // Сгенерированный уникальный GUID для нашего бэкенда
    static ggml_guid guid = { 0x72, 0x6b, 0x6e, 0x70, 0x75, 0x32, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01 };
    return &guid;
}

static void ggml_backend_rknpu2_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rknpu2_device_get_name(dev);
    props->description = ggml_backend_rknpu2_device_get_description(dev);
    props->type        = ggml_backend_rknpu2_device_get_type(dev);
    // Мы не можем легко получить объем памяти NPU, поэтому оставляем нули
    props->memory_free  = 0;
    props->memory_total = 0;
    // Возможности нашего бэкенда
    props->caps = (struct ggml_backend_dev_caps){
        /* .async                 = */ false, // Мы работаем синхронно
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false, // Мы не можем использовать память CPU напрямую
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_rknpu2_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    auto * ctx = new ggml_backend_rknpu2_context();
    // TODO: parse params to set ctx->core_mask
    ggml_backend_t backend = new ggml_backend{
        /* .guid    = */ ggml_backend_rknpu2_guid(),
        /* .iface   = */ rknpu2_backend_interface,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
    return backend;
}

static void ggml_backend_rknpu2_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // RKNPU CMA память управляется ядром, точные цифры получить сложно.
    // Возвращаем нули, это безопасно.
    (void)dev;
    *free = 0;
    *total = 0;
}

static ggml_backend_buffer_type_t ggml_backend_rknpu2_device_get_buffer_type(ggml_backend_dev_t dev) {
    static struct ggml_backend_buffer_type rknpu2_buffer_type = {
        /* .iface   = */ rknpu2_buffer_type_interface,
        /* .device  = */ dev,
        /* .context = */ nullptr,
    };
    return &rknpu2_buffer_type;
}

static bool ggml_backend_rknpu2_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // Мы поддерживаем только наш собственный тип буфера
    return buft == ggml_backend_rknpu2_device_get_buffer_type(dev);
}

static bool ggml_backend_rknpu2_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    (void)dev;
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_CPY:
            return true;

        case GGML_OP_MUL_MAT: {
            const auto * src0 = op->src[0]; // веса
            const auto * src1 = op->src[1]; // активации

            // Проверяем типы
            if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32) {
                return false;
            }

            // Проверяем размерности весов (src0)
            if (ggml_n_dims(src0) != 2) {
                return false;
            }
            
            const int64_t n = src0->ne[0];
            const int64_t k = src0->ne[1];

            // Проверяем аппаратные ограничения RKNPU2
            if (k % 32 != 0 || n % 32 != 0) {
                return false;
            }

            return true; // Мы можем выполнить эту операцию
        }
        default: return false;
    }
}

static bool ggml_backend_rknpu2_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    // Эта функция говорит llama.cpp, что мы хотим выполнить эту операцию,
    // даже если ее входные тензоры находятся в памяти CPU.
    return ggml_backend_rknpu2_device_supports_op(dev, op);
}

static const struct ggml_backend_device_i rknpu2_device_interface = {
    /* .get_name             = */ ggml_backend_rknpu2_device_get_name,
    /* .get_description      = */ ggml_backend_rknpu2_device_get_description,
    /* .get_memory           = */ ggml_backend_rknpu2_device_get_memory,
    /* .get_type             = */ ggml_backend_rknpu2_device_get_type,
    /* .get_props            = */ ggml_backend_rknpu2_device_get_props,
    /* .init_backend         = */ ggml_backend_rknpu2_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_rknpu2_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_rknpu2_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rknpu2_device_supports_buft,
    /* .offload_op           = */ ggml_backend_rknpu2_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// --- Registration Interface ---

static const char * ggml_backend_rknpu2_reg_get_name(ggml_backend_reg_t reg) {
    return "RKNPU2";
}

static size_t ggml_backend_rknpu2_reg_get_device_count(ggml_backend_reg_t reg) {
    // У нас одно NPU устройство, но оно может иметь несколько ядер
    return 1;
}

static ggml_backend_dev_t ggml_backend_rknpu2_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static struct ggml_backend_device rknpu2_device = {
        /* .iface   = */ rknpu2_device_interface,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    return &rknpu2_device;
}

static const struct ggml_backend_reg_i rknpu2_reg_interface = {
    /* .get_name         = */ ggml_backend_rknpu2_reg_get_name,
    /* .get_device_count = */ ggml_backend_rknpu2_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rknpu2_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

ggml_backend_reg_t ggml_backend_rknpu2_reg(void) {
    static struct ggml_backend_reg rknpu2_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ rknpu2_reg_interface,
        /* .context     = */ nullptr,
    };
    return &rknpu2_reg;
}