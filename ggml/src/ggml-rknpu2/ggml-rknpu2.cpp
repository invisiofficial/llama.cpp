#include "ggml-backend-impl.h"
#include "ggml-rknpu2.h"
#include "ggml-impl.h"

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

static struct ggml_rknpu2_matmul_kernel* ggml_rknpu2_matmul_kernel_find(int m, int k, int n, rknn_tensor_type type) {
    for (int i = 0; i < matmul_kernels_count; i++) {
        struct ggml_rknpu2_matmul_kernel* kernel = &matmul_kernels[i];
        if (kernel->matmul_info.M == m && kernel->matmul_info.K == k && kernel->matmul_info.N == n &&
            type == kernel->matmul_info.type)
            return kernel;
    }
    return NULL;
}

static struct ggml_rknpu2_matmul_kernel* ggml_rknpu2_matmul_kernel_create(int m, int k, int n, rknn_tensor_type type) {
    struct ggml_rknpu2_matmul_kernel* kernel = ggml_rknpu2_matmul_kernel_find(m, k, n, type);
    if (kernel != NULL) return kernel;

    GGML_ASSERT(matmul_kernels_count < GGML_RKNPU2_MAX_MATMUL_KERNELS);
    kernel = &matmul_kernels[matmul_kernels_count++];
    memset(kernel, 0, sizeof(struct ggml_rknpu2_matmul_kernel));

    kernel->matmul_info.M = m;
    kernel->matmul_info.K = k;
    kernel->matmul_info.N = n;
    kernel->matmul_info.type = type;
    kernel->matmul_info.native_layout = 1;
    kernel->matmul_info.perf_layout = 0;

    int ret = rknn_matmul_create(&kernel->matmul_ctx, &kernel->matmul_info, &kernel->matmul_io_attr);
    GGML_ASSERT(ret == 0);
    // TODO: Make core mask configurable
    rknn_matmul_set_core_mask(kernel->matmul_ctx, RKNN_NPU_CORE_0_1_2);
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
static void ggml_rknpu2_transposed_to_native_int8(int8_t * dst, const float * src, size_t k, size_t n) {
    GGML_ASSERT(k % 32 == 0 && n % 32 == 0 && k > 0 && n > 0);
    const size_t rknpu_strides[4] = {k / 32 * 32 * 32, 32 * 32, 32, 1};
    for (size_t j = 0; j < k / 32; j++) {
        for (size_t i = 0; i < n / 32; i++) {
            for (size_t ii = 0; ii < 32; ii++) {
                for (size_t jj = 0; jj < 32; jj++) {
                    size_t src_idx = j * 32 + (i * 32 + ii) * k + jj;
                    size_t dst_idx = i * rknpu_strides[0] + j * rknpu_strides[1] + ii * rknpu_strides[2] + jj;
                    dst[dst_idx] = roundf(fminf(fmaxf(src[src_idx], -1.0f), 1.0f) * 127.0f);
                }
            }
        }
    }
}


//================================================================================
// GGML Backend Implementation
//================================================================================

// Backend context
struct ggml_backend_rknpu2_context {
    int core_mask = RKNN_NPU_CORE_0_1_2;
};

// Buffer context (stores info about a DMA allocation)
struct ggml_backend_rknpu2_buffer_context {
    int fd = -1;
    void * va = nullptr;
    size_t size = 0;
};

// Tensor extra data (stores prepared weights)
struct ggml_backend_rknpu2_tensor_extra {
    rknn_tensor_mem* b_mem;
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
    /* .init_tensor    = */ ggml_backend_rknpu2_buffer_init_tensor,
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
            const auto * src0 = node->src[0]; // веса (на NPU)
            const auto * src1 = node->src[1]; // активации (на CPU)
            auto * dst = node;
            
            // Ленивая инициализация весов
            if (src0->extra == nullptr) {
                // Этот тензор весов мы видим впервые. Нужно его подготовить.
                const int64_t k = src0->ne[1];
                const int64_t n = src0->ne[0];
                
                // 1. Деквантизация Q8_0 -> F32
                size_t nelements = ggml_nelements(src0);
                std::vector<float> fdata(nelements);
                ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(src0->data, fdata.data(), nelements);

                // 2. Трансформация в нативный INT8 формат RKNPU
                std::vector<int8_t> reordered_data(nelements);
                ggml_rknpu2_transposed_to_native_int8(reordered_data.data(), fdata.data(), k, n);
                
                // 3. Создаем ядро, чтобы получить matmul_ctx
                struct ggml_rknpu2_matmul_kernel* kernel_init = ggml_rknpu2_matmul_kernel_create(1, k, n, RKNN_TENSOR_INT8);

                // 4. Создаем rknn_tensor_mem, но не из файла, а из нашего же буфера, где уже лежат веса
                ggml_backend_buffer_t buffer = src0->buffer;
                auto * buffer_ctx = (ggml_backend_rknpu2_buffer_context *)buffer->context;
                size_t tensor_offset = (uintptr_t)src0->data - (uintptr_t)buffer_ctx->va;
                
                rknn_tensor_mem* b_mem = rknn_create_mem_from_fd(kernel_init->matmul_ctx, buffer_ctx->fd, buffer_ctx->va, kernel_init->matmul_io_attr.B.size, tensor_offset);

                // 5. Копируем преобразованные данные в DMA-память (поверх старых)
                memcpy(b_mem->virt_addr, reordered_data.data(), kernel_init->matmul_io_attr.B.size);

                // 6. Сохраняем указатель
                src0->extra = new ggml_backend_rknpu2_tensor_extra{b_mem};
            }

            GGML_ASSERT(src0->extra != nullptr && "RKNPU2: weight tensor not prepared");
            auto * tensor_extra = (ggml_backend_rknpu2_tensor_extra*) src0->extra;

            const int64_t m = src1->ne[1];
            const int64_t k = src0->ne[1];
            const int64_t n = src0->ne[0];

            struct ggml_rknpu2_matmul_kernel* kernel = ggml_rknpu2_matmul_kernel_create(m, k, n, RKNN_TENSOR_INT8);
            GGML_ASSERT(kernel != nullptr);

            // Подготовка активаций (матрица A): F32 -> INT8
            const float * src1_data = (const float *) src1->data;
            int8_t * a_virt = (int8_t *) kernel->A->virt_addr;
            for (size_t j = 0; j < (size_t)m * k; j++) {
                float val = roundf(fminf(fmaxf(src1_data[j] * 127.0f / GGML_RKNPU2_INPUT_SCALE, -127.0f), 127.0f));
                a_virt[j] = (int8_t)val;
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
            for (size_t j = 0; j < (size_t)m * n; j++) {
                dst_data[j] = (float)c_virt[j] / 127.0f / 127.0f * GGML_RKNPU2_INPUT_SCALE;
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
            const auto * src0 = op->src[0]; const auto * src1 = op->src[1];
            return src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32 &&
                   src0->ne[0] % 32 == 0 && src0->ne[1] % 32 == 0;
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