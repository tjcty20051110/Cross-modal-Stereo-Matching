// conv_bn_relu.h — Basic ConvBnReLU building block shared across CFM modules.
#pragma once

#ifdef CMS_HAS_LIBTORCH

#include <torch/torch.h>

namespace cms {
namespace cfm {

class ConvBnReLUImpl : public torch::nn::Module {
public:
    ConvBnReLUImpl(int in_c, int out_c, int k = 3, int s = 1, int p = 1);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv_{nullptr};
    torch::nn::BatchNorm2d bn_{nullptr};
};
TORCH_MODULE(ConvBnReLU);

}  // namespace cfm
}  // namespace cms

#endif  // CMS_HAS_LIBTORCH
