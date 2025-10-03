#include "ggml-rknpu2.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"

#include <cassert>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

// RKNPU2 SDK headers
#include <rknn_api.h>
#include <rknn_matmul_api.hh>

// For DMA buffer management
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <arm_neon.h>

#define GGML_RKNPU2_INPUT_SCALE 1.7f

//==================================================================================================
// Low-level DMA and helpers (from old implementation)
//==================================================================================================

struct dma_heap_allocation_data {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};

#define DMA_HEAP_IOC_MAGIC      'H'
#define DMA_HEAP_IOCTL_ALLOC    _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct dma_heap_allocation_data)
#define DMA_BUF_SYNC_READ       (1 << 0)
#define DMA_BUF_SYNC_WRITE      (2 << 0)
#define DMA_BUF_SYNC_RW         (DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE)
#define DMA_BUF_SYNC_START      (0 << 2)
#define DMA_BUF_SYNC_END        (1 << 2)
#define DMA_BUF_BASE            'b'
#define DMA_BUF_IOCTL_SYNC      _IOW(DMA_BUF_BASE, 0, uint64_t)

static int dma_alloc(size_t size, int *fd, void **va) {
    int ret;
    int prot;
    void *mmap_va;
    int dma_heap_fd = -1;
    struct dma_heap_allocation_data buf_data;
    const char* path = "/dev/dma_heap/system";

    dma_heap_fd = open(path, O_RDWR);
    if (dma_heap_fd < 0) {
        GGML_LOG_ERROR("open %s fail!\n", path);
        return dma_heap_fd;
    }

    memset(&buf_data, 0x0, sizeof(struct dma_heap_allocation_data));
    buf_data.len = size;
    buf_data.fd_flags = O_CLOEXEC | O_RDWR;
    ret = ioctl(dma_heap_fd, DMA_HEAP_IOCTL_ALLOC, &buf_data);
    if (ret < 0) {
        GGML_LOG_ERROR("RK_DMA_HEAP_ALLOC_BUFFER failed\n");
        close(dma_heap_fd);
        return ret;
    }

    prot = PROT_READ | PROT_WRITE;
    mmap_va = (void *)mmap(NULL, buf_data.len, prot, MAP_SHARED, buf_data.fd, 0);
    if (mmap_va == MAP_FAILED) {
        GGML_LOG_ERROR("mmap failed: %s\n", strerror(errno));
        close(buf_data.fd);
        close(dma_heap_fd);
        return -errno;
    }

    *va = mmap_va;
    *fd = buf_data.fd;
    close(dma_heap_fd);
    return 0;
}

static void dma_buf_free(size_t size, int *fd, void *va) {
    if (va) munmap(va, size);
    if (*fd >= 0) close(*fd);
    *fd = -1;
}

static int dma_sync_cpu_to_device(int fd) {
    uint64_t flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW;
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &flags);
}

static int dma_sync_device_to_cpu(int fd) {
    uint64_t flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW;
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &flags);
}

static __fp16 arm_fp32_to_fp16(float x) {
    return (__fp16)x;
}

//==================================================================================================
// RKNPU2 MatMul Kernel Management
//==================================================================================================

struct ggml_rknpu2_matmul_kernel {
    rknn_matmul_info matmul_info;
    rknn_matmul_ctx matmul_ctx;
    rknn_matmul_io_attr matmul_io_attr;

    rknn_tensor_mem* A; // Input activations
    rknn_tensor_mem* C; // Output
};

#define GGML_RKNPU2_MAX_MATMUL_KERNELS 64
static std::vector<ggml_rknpu2_matmul_kernel> matmul_kernels;

static ggml_rknpu2_matmul_kernel * ggml_rknpu2_matmul_kernel_find(int m, int k, int n) {
    for (auto & kernel : matmul_kernels) {
        if (kernel.matmul_info.M == m && kernel.matmul_info.K == k && kernel.matmul_info.N == n) {
            return &kernel;
        }
    }
    return nullptr;
}

static ggml_rknpu2_matmul_kernel* ggml_rknpu2_matmul_kernel_create(int m, int k, int n) {
    if (ggml_rknpu2_matmul_kernel_find(m, k, n)) {
        return ggml_rknpu2_matmul_kernel_find(m, k, n);
    }

    GGML_ASSERT(matmul_kernels.size() < GGML_RKNPU2_MAX_MATMUL_KERNELS);
    
    ggml_rknpu2_matmul_kernel kernel;
    memset(&kernel, 0, sizeof(ggml_rknpu2_matmul_kernel));

    kernel.matmul_info.M = m;
    kernel.matmul_info.K = k;
    kernel.matmul_info.N = n;
    kernel.matmul_info.type = RKNN_INT8_MM_INT8_TO_INT32;
    kernel.matmul_info.native_layout = 1; // B (weights) use native layout
    kernel.matmul_info.perf_layout = 0; // A and C use original layout

    int ret = rknn_matmul_create(&kernel.matmul_ctx, &kernel.matmul_info, &kernel.matmul_io_attr);
    GGML_ASSERT(ret == 0);

    // Use only one NPU core for now. TODO: Explore multi-core usage.
    rknn_matmul_set_core_mask(kernel.matmul_ctx, RKNN_NPU_CORE_0);

    GGML_LOG_INFO("Created RKNPU2 matmul kernel: (%d, %d) x (%d, %d) = (%d, %d)\n", m, k, k, n, m, n);
    
    // Allocate memory for A and C which are reused across inferences
    kernel.A = rknn_create_mem(kernel.matmul_ctx, kernel.matmul_io_attr.A.size);
    kernel.C = rknn_create_mem(kernel.matmul_ctx, kernel.matmul_io_attr.C.size);

    rknn_matmul_set_io_mem(kernel.matmul_ctx, kernel.A, &kernel.matmul_io_attr.A);
    rknn_matmul_set_io_mem(kernel.matmul_ctx, kernel.C, &kernel.matmul_io_attr.C);
    
    matmul_kernels.push_back(kernel);
    return &matmul_kernels.back();
}

static void ggml_rknpu2_destroy() {
    for (auto & kernel : matmul_kernels) {
        rknn_destroy_mem(kernel.matmul_ctx, kernel.A);
        rknn_destroy_mem(kernel.matmul_ctx, kernel.C);
        rknn_matmul_destroy(kernel.matmul_ctx);
    }
    matmul_kernels.clear();
}


//==================================================================================================
// Backend-specific data attached to ggml_tensor
//==================================================================================================

struct ggml_rknpu2_data_pack {
    void* ordered_data; // Temporary host buffer for transformed weights before moving to DMA
    bool initialized;   // Flag to indicate if rknn_tensor_mem for weights is created

    // RKNPU2 API structs
    rknn_tensor_mem* B; // Weights in DMA memory
};

static void ggml_rknpu2_free_data(struct ggml_tensor * tensor) {
    if (tensor->extra == nullptr) {
        return;
    }

    ggml_rknpu2_data_pack* pack = (ggml_rknpu2_data_pack*)tensor->extra;
    
    if (pack->ordered_data) {
        free(pack->ordered_data);
    }
    if (pack->initialized && pack->B) {
        // HACK: Grab a kernel context to release the memory. This assumes all kernels share a compatible context.
        GGML_ASSERT(!matmul_kernels.empty());
        rknn_destroy_mem(matmul_kernels[0].matmul_ctx, pack->B);
    }
    
    delete pack;
    tensor->extra = nullptr;
}


//==================================================================================================
// ggml-backend buffer interface
//==================================================================================================

struct ggml_backend_rknpu2_buffer_context {
    int fd = -1;
    void * va = nullptr;
    size_t size = 0;
};

static void ggml_backend_rknpu2_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu2_buffer_context * ctx = (ggml_backend_rknpu2_buffer_context *)buffer->context;
    dma_buf_free(ctx->size, &ctx->fd, ctx->va);
    delete ctx;
}

static void * ggml_backend_rknpu2_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu2_buffer_context * ctx = (ggml_backend_rknpu2_buffer_context *)buffer->context;
    return ctx->va;
}

static void ggml_backend_rknpu2_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor->buffer->iface.get_base(tensor->buffer) != nullptr && "buffer has no base");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "offset + size > ggml_nbytes(tensor)");

    ggml_backend_rknpu2_buffer_context * ctx = (ggml_backend_rknpu2_buffer_context *)buffer->context;
    
    memset((uint8_t *)tensor->data + offset, value, size);
    dma_sync_cpu_to_device(ctx->fd);
}

static void ggml_backend_rknpu2_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor->buffer->iface.get_base(tensor->buffer) != nullptr && "buffer has no base");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "offset + size > ggml_nbytes(tensor)");
    
    ggml_backend_rknpu2_buffer_context * ctx = (ggml_backend_rknpu2_buffer_context *)buffer->context;
    
    memcpy((uint8_t *)tensor->data + offset, data, size);
    dma_sync_cpu_to_device(ctx->fd);
}

static void ggml_backend_rknpu2_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor->buffer->iface.get_base(tensor->buffer) != nullptr && "buffer has no base");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "offset + size > ggml_nbytes(tensor)");

    ggml_backend_rknpu2_buffer_context * ctx = (ggml_backend_rknpu2_buffer_context *)buffer->context;

    dma_sync_device_to_cpu(ctx->fd);
    memcpy(data, (const uint8_t *)tensor->data + offset, size);
}

static void ggml_backend_rknpu2_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rknpu2_buffer_context * ctx = (ggml_backend_rknpu2_buffer_context *)buffer->context;
    memset(ctx->va, value, ctx->size);
    dma_sync_cpu_to_device(ctx->fd);
}

// Forward declaration
static void ggml_rknpu2_transform_tensor(void * data, struct ggml_tensor * tensor);

static enum ggml_status ggml_backend_rknpu2_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    // This function is called when a tensor is allocated in this buffer.
    // This is the perfect place to "prepare" the tensor for the backend.
    
    if (tensor->op == GGML_OP_NONE && ggml_is_quantized(tensor->type)) {
         GGML_LOG_DEBUG("%s: preparing weight tensor '%s' for RKNPU2\n", __func__, tensor->name);
         // It's a weight tensor, transform it.
         ggml_rknpu2_transform_tensor(tensor->data, tensor);
    }
    
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rknpu2_buffer_reset(ggml_backend_buffer_t buffer) {
    // This is called when the buffer is reset. We should free any backend-specific data.
    // The scheduler keeps track of tensors in the buffer, we can iterate them.
    for (size_t i = 0; i < buffer->n_tensors; i++) {
        struct ggml_tensor * tensor = buffer->tensors[i];
        if (tensor->extra != nullptr) {
            ggml_rknpu2_free_data(tensor);
        }
    }
}

static struct ggml_backend_buffer_i ggml_backend_rknpu2_buffer_interface = {
    /* .free_buffer   = */ ggml_backend_rknpu2_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_rknpu2_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_rknpu2_buffer_init_tensor,
    /* .memset_tensor = */ ggml_backend_rknpu2_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_rknpu2_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_rknpu2_buffer_get_tensor,
    /* .cpy_tensor    = */ NULL, // use default copy
    /* .clear         = */ ggml_backend_rknpu2_buffer_clear,
    /* .reset         = */ ggml_backend_rknpu2_buffer_reset,
};

//==================================================================================================
// ggml-backend buffer type interface
//==================================================================================================

static const char * ggml_backend_rknpu2_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "RKNPU2";
}

static ggml_backend_buffer_t ggml_backend_rknpu2_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rknpu2_buffer_context * ctx = new ggml_backend_rknpu2_buffer_context;
    
    if (dma_alloc(size, &ctx->fd, &ctx->va) != 0) {
        GGML_LOG_ERROR("%s: failed to allocate DMA buffer of size %zu\n", __func__, size);
        delete ctx;
        return nullptr;
    }
    
    ctx->size = size;
    GGML_LOG_INFO("%s: allocated DMA buffer of size %zu MB\n", __func__, size / (1024*1024));
    
    return ggml_backend_buffer_init(buft, ggml_backend_rknpu2_buffer_interface, ctx, size);
}

static size_t ggml_backend_rknpu2_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    // RKNPU requires 32-byte alignment for dimensions
    return 32;
}

//==================================================================================================
// ggml-backend (stream) interface
//==================================================================================================

static const char * ggml_backend_rknpu2_get_name(ggml_backend_t backend) {
    return "RKNPU2";
}

static void ggml_backend_rknpu2_free(ggml_backend_t backend) {
    // The backend context is just a placeholder in this implementation
    delete (char *)backend->context;
    delete backend;
}

static enum ggml_status ggml_backend_rknpu2_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        
        if (node->op_backend != backend) {
            continue;
        }

        if (node->op == GGML_OP_MUL_MAT) {
            struct ggml_tensor * src0 = node->src[0];
            struct ggml_tensor * src1 = node->src[1];
            struct ggml_tensor * dst  = node;
            
            GGML_ASSERT(src0->extra != nullptr && "RKNPU2: weight tensor not prepared");
            
            ggml_rknpu2_data_pack* pack = (ggml_rknpu2_data_pack*)src0->extra;

            const int64_t m = src1->ne[1];
            const int64_t k = src0->ne[0];
            const int64_t n = dst->ne[0];

            ggml_rknpu2_matmul_kernel* kernel = ggml_rknpu2_matmul_kernel_create(m, k, n);
            GGML_ASSERT(kernel != nullptr);

            // First time this weight tensor is used for computation
            if (!pack->initialized) {
                GGML_LOG_DEBUG("RKNPU2: initializing weight tensor '%s' on NPU\n", src0->name);
                int fd = -1;
                uint8_t *va = NULL;
                dma_alloc(kernel->matmul_io_attr.B.size, &fd, (void **)&va);
                pack->B = rknn_create_mem_from_fd(kernel->matmul_ctx, fd, va, kernel->matmul_io_attr.B.size, 0);
                
                memcpy(pack->B->virt_addr, pack->ordered_data, kernel->matmul_io_attr.B.size);
                dma_sync_cpu_to_device(fd);
                
                free(pack->ordered_data);
                pack->ordered_data = nullptr;
                pack->initialized = true;
            }
            
            // A: fp32 -> int8 (Input activations)
            float const* src1_data = (float const*)src1->data;
            int8_t* A = (int8_t*)kernel->A->virt_addr;
            for(size_t j = 0; j < m*k; j++) {
                A[j] = round(fmin(fmax(src1_data[j]*127.0f/GGML_RKNPU2_INPUT_SCALE, -127.0f), 127.0f));
            }

            int ret;
            ret = rknn_matmul_set_io_mem(kernel->matmul_ctx, pack->B, &kernel->matmul_io_attr.B);
            GGML_ASSERT(ret == 0);
            ret = rknn_matmul_run(kernel->matmul_ctx);
            GGML_ASSERT(ret == 0);

            // C: int32 -> fp32 (Output)
            float* dst_data = (float*)dst->data;
            int32_t* C = (int32_t*)kernel->C->virt_addr;
            for(size_t j = 0; j < m*n; j++) {
                dst_data[j] = C[j] / 127.0f / 127.0f * GGML_RKNPU2_INPUT_SCALE;
            }
        }
    }
    return GGML_STATUS_SUCCESS;
}

static bool ggml_backend_rknpu2_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];
            
            // RKNPU2 only supports Q8_0 weights and F32 activations
            if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32) {
                return false;
            }
            
            // Check alignment constraints
            const int64_t k = src0->ne[0];
            const int64_t n = src0->ne[1];
            if (k % 32 != 0 || n % 32 != 0) {
                return false;
            }
            
            // Check dimension limits (from old code)
            if (k > 10240 || n > 4096) {
                return false;
            }
            return true;
        }
        // The scheduler may assign view ops to this backend, we should accept them
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
            return true;
        default:
            return false;
    }
}

//==================================================================================================
// ggml-backend device interface
//==================================================================================================

static const char * ggml_backend_rknpu2_device_get_name(ggml_backend_dev_t dev) {
    return "RKNPU2";
}

static const char * ggml_backend_rknpu2_device_get_description(ggml_backend_dev_t dev) {
    return "Rockchip NPU";
}

static enum ggml_backend_dev_type ggml_backend_rknpu2_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_NPU;
}

static void ggml_backend_rknpu2_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // RKNPU uses system memory via DMA, so we can report a large portion of system memory.
    // This is a rough estimate.
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages * page_size;
    
    // Reporting free memory is tricky. Let's report half of total as a heuristic.
    *free = *total / 2;
}

static void ggml_backend_rknpu2_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->type = ggml_backend_rknpu2_device_get_type(dev);
    props->has_blas = false;
    props->has_matrix = true; // For MatMul
    props->max_graph_size = 1024; // A reasonable default
    props->max_nodes = 4096; // A reasonable default
    ggml_backend_rknpu2_device_get_memory(dev, &props->mem_free, &props->mem_total);
    props->numa_node = 0;
}

static ggml_backend_buffer_type_t ggml_backend_rknpu2_device_get_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_t ggml_backend_rknpu2_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_t backend = new ggml_backend;
    
    *backend = {
        /* .guid   = */ {0}, // Will be generated
        /* .iface  = */ {
            /* .get_name         = */ ggml_backend_rknpu2_get_name,
            /* .free             = */ ggml_backend_rknpu2_free,
            /* .set_tensor_async = */ NULL,
            /* .get_tensor_async = */ NULL,
            /* .cpy_tensor_async = */ NULL,
            /* .synchronize      = */ NULL,
            /* .graph_plan_create = */ NULL,
            /* .graph_plan_free   = */ NULL,
            /* .graph_plan_update = */ NULL,
            /* .graph_plan_compute= */ NULL,
            /* .graph_compute    = */ ggml_backend_rknpu2_graph_compute,
            /* .supports_op      = */ NULL, // Deprecated, use device supports_op
            /* .offload_op       = */ NULL, // Deprecated, use device offload_op
        },
        /* .device = */ dev,
        /* .context = */ new char, // Placeholder context
    };
    
    return backend;
}

static struct ggml_backend_device_i ggml_backend_rknpu2_device_interface = {
    /* .get_name             = */ ggml_backend_rknpu2_device_get_name,
    /* .get_description      = */ ggml_backend_rknpu2_device_get_description,
    /* .get_memory           = */ ggml_backend_rknpu2_device_get_memory,
    /* .get_type             = */ ggml_backend_rknpu2_device_get_type,
    /* .get_props            = */ ggml_backend_rknpu2_device_get_props,
    /* .init_backend         = */ ggml_backend_rknpu2_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_rknpu2_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_rknpu2_supports_op,
    /* .supports_buft        = */ NULL, // Supports its own buffer type by default
    /* .offload_op           = */ ggml_backend_rknpu2_supports_op, // Use same logic for offloading
};

static ggml_backend_device ggml_backend_rknpu2_device = {
    /* .iface   = */ ggml_backend_rknpu2_device_interface,
    /* .reg     = */ NULL, // Will be set by registration
    /* .context = */ NULL,
};

static ggml_backend_buffer_type_t ggml_backend_rknpu2_device_get_buffer_type(ggml_backend_dev_t dev) {
    static struct ggml_backend_buffer_type ggml_backend_rknpu2_buffer_type = {
        /* .iface = */ {
            /* .get_name       = */ ggml_backend_rknpu2_buffer_type_get_name,
            /* .alloc_buffer   = */ ggml_backend_rknpu2_buffer_type_alloc_buffer,
            /* .get_alignment  = */ ggml_backend_rknpu2_buffer_type_get_alignment,
            /* .get_max_size   = */ NULL, // No max size
            /* .get_alloc_size = */ NULL, // Default size
            /* .is_host        = */ NULL, // Not host memory
        },
        /* .device = */ &ggml_backend_rknpu2_device,
        /* .context = */ NULL,
    };
    return &ggml_backend_rknpu2_buffer_type;
}

//==================================================================================================
// Backend registration
//==================================================================================================

static const char * ggml_backend_rknpu2_reg_get_name(ggml_backend_reg_t reg) {
    return "RKNPU2";
}

static size_t ggml_backend_rknpu2_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;
}

static ggml_backend_dev_t ggml_backend_rknpu2_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    return &ggml_backend_rknpu2_device;
}

GGML_API ggml_backend_reg_t ggml_backend_rknpu2_reg(void) {
    static struct ggml_backend_reg rknpu2_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface = */ {
            /* .get_name          = */ ggml_backend_rknpu2_reg_get_name,
            /* .get_device_count = */ ggml_backend_rknpu2_reg_get_device_count,
            /* .get_device       = */ ggml_backend_rknpu2_reg_get_device,
        },
        /* .context = */ NULL,
    };
    
    // Set the registration for the device
    ggml_backend_rknpu2_device.reg = &rknpu2_reg;
    
    // One-time initialization for the backend
    // For example, initializing the kernel pool
    atexit(ggml_rknpu2_destroy);
    
    return &rknpu2_reg;
}


//==================================================================================================
// Tensor transformation (ported from old implementation)
//==================================================================================================
static void ggml_rknpu2_transposed_to_native_int8(int8_t *restrict dst,
                                                  const float *restrict src,
                                                  size_t k, size_t n) {
    GGML_ASSERT(k % 32 == 0 && n % 32 == 0 && k > 0 && n > 0);

    // RKNN native layout is (N/32, K/32, 32, 32)
    const size_t rknpu_strides[4] = {k / 32 * 32 * 32, 32 * 32, 32, 1};

    for (size_t j = 0; j < k / 32; j++) {
        for (size_t i = 0; i < n / 32; i++) {
            for (size_t ii = 0; ii < 32; ii++) {
                size_t partial_src_idx = j * 32 + (i * 32 + ii) * k;
                size_t partial_dst_idx = i * rknpu_strides[0] + j * rknpu_strides[1] + ii * rknpu_strides[2];
                for (size_t jj = 0; jj < 32; jj++) {
                    size_t src_idx = partial_src_idx + jj;
                    size_t dst_idx = partial_dst_idx + jj;
                    // Note: old code used fmin/fmax(src, -1, 1) * 127. Here we assume src is already scaled.
                    // This is consistent with how ggml dequantizes Q8_0.
                    dst[dst_idx] = round(src[src_idx] * 127.0f);
                }
            }
        }
    }
}

void ggml_rknpu2_transform_tensor(void * data, struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(ggml_is_quantized(tensor->type));

    const int64_t ne0 = tensor->ne[0]; // k
    const int64_t ne1 = tensor->ne[1]; // n
    const size_t nelements = ne0 * ne1;

    ggml_type_traits_t traits = ggml_get_type_traits(tensor->type);
    GGML_ASSERT(traits.to_float != NULL);

    float* fdata = new float[nelements];
    traits.to_float(data, fdata, nelements);

    int8_t* reordered_data = new int8_t[nelements];
    // Note: The old implementation swapped ne0 and ne1 here. This seems to be because
    // ggml tensors are row-major (k, n), but the transformation function expects (n, k) layout
    // for a transposed matrix. Here, we pass k=ne0, n=ne1.
    ggml_rknpu2_transposed_to_native_int8(reordered_data, fdata, ne0, ne1);

    delete[] fdata;
    
    ggml_rknpu2_data_pack* pack = new ggml_rknpu2_data_pack;
    memset(pack, 0, sizeof(ggml_rknpu2_data_pack));

    pack->ordered_data = reordered_data;
    pack->initialized = false;
    pack->B = nullptr;

    tensor->extra = pack;
    // The original tensor->data is now owned by the backend. We nullify it to prevent double free.
    // The data is dequantized into fdata and then reordered into reordered_data which is attached to extra.
    // The original data is no longer needed in its ggml format.
    // However, the `init_tensor` call happens after data has been copied from the file into the buffer.
    // We should not modify tensor->data itself as it's just a pointer into the DMA buffer.
}