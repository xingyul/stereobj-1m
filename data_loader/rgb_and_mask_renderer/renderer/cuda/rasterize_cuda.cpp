#include <torch/torch.h>

#include <tuple>

// CUDA forward declarations

std::tuple<at::Tensor, at::Tensor, at::Tensor> forward_cuda(
        const at::Tensor& faces,
        const at::Tensor& textures,
        const int image_height,
        const int image_width,
        const float near,
        const float far);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::tuple<at::Tensor, at::Tensor, at::Tensor> forward_cpu(
        const at::Tensor& faces,
        const at::Tensor& textures,
        const int image_height,
        const int image_width,
        const float near,
        const float far) {

    CHECK_INPUT(faces);
    CHECK_INPUT(textures);

    return forward_cuda(faces, textures, image_height, image_width, near, far);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "FORWARD (CUDA)");
}
