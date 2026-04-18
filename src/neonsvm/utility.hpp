#ifndef _NEONSVM_UTILITY_HPP_
#define _NEONSVM_UTILITY_HPP_

#include <arm_neon.h>
#include <cstddef>
#include <stdexcept>
#include <vector>

#define NOT_IMPLEMENTED throw std::logic_error("Not implemented yet")

namespace neonsvm::utility {
  // https://github.com/Tencent/ncnn/blob/426cf6673242e4eecf6791d720bd7a7b9b59cdb3/src/layer/arm/neon_mathfun.h#L132
  inline float32x4_t vexpq_f32(float32x4_t x) {
    constexpr float c_exp_hi = 88.3762626647949f;
    constexpr float c_exp_lo = -88.3762626647949f;

    constexpr float c_cephes_LOG2EF = 1.44269504088896341;
    constexpr float c_cephes_exp_C1 = 0.693359375;
    constexpr float c_cephes_exp_C2 = -2.12194440e-4;

    constexpr float c_cephes_exp_p0 = 1.9875691500E-4;
    constexpr float c_cephes_exp_p1 = 1.3981999507E-3;
    constexpr float c_cephes_exp_p2 = 8.3334519073E-3;
    constexpr float c_cephes_exp_p3 = 4.1665795894E-2;
    constexpr float c_cephes_exp_p4 = 1.6666665459E-1;
    constexpr float c_cephes_exp_p5 = 5.0000001201E-1;

    float32x4_t one = vdupq_n_f32(1);
    x = vminq_f32(x, vdupq_n_f32(c_exp_hi));
    x = vmaxq_f32(x, vdupq_n_f32(c_exp_lo));

    /* express exp(x) as exp(g + n*log(2)) */
    float32x4_t fx = vfmaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(c_cephes_LOG2EF));

    /* perform a floorf */
    float32x4_t tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));

    /* if greater, substract 1 */
    uint32x4_t mask = vcgtq_f32(tmp, fx);
    mask = vandq_u32(mask, vreinterpretq_u32_f32(one));

    fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));

    tmp = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C1));
    float32x4_t z = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C2));
    x = vsubq_f32(x, tmp);
    x = vsubq_f32(x, z);

    z = vmulq_f32(x, x);

    float32x4_t y = vdupq_n_f32(c_cephes_exp_p0);
    y = vfmaq_f32(vdupq_n_f32(c_cephes_exp_p1), y, x);
    y = vfmaq_f32(vdupq_n_f32(c_cephes_exp_p2), y, x);
    y = vfmaq_f32(vdupq_n_f32(c_cephes_exp_p3), y, x);
    y = vfmaq_f32(vdupq_n_f32(c_cephes_exp_p4), y, x);
    y = vfmaq_f32(vdupq_n_f32(c_cephes_exp_p5), y, x);

    y = vfmaq_f32(x, y, z);
    y = vaddq_f32(y, one);

    /* build 2^n */
    int32x4_t mm = vcvtq_s32_f32(fx);
    mm = vaddq_s32(mm, vdupq_n_s32(0x7f));
    mm = vshlq_n_s32(mm, 23);
    float32x4_t pow2n = vreinterpretq_f32_s32(mm);

    y = vmulq_f32(y, pow2n);
    return y;
  }

  constexpr size_t align_size(size_t size, size_t alignment) {
    return (size % alignment == 0) ? size : size + alignment - (size % alignment);
  }

  template <typename T>
  constexpr T clamp(T value, T min, T max) {
    return std::max(min, std::min(value, max));
  }

  constexpr bool equal_approx(float first, float second, float error = 1e-3f) {
    return std::abs(first - second) <= error;
  }

  inline bool equal_approx(const std::vector<float>& first, const std::vector<float>& second, float error = 1e-3f) {
    if (first.size() != second.size()) return false;

    for (size_t i = 0; i < first.size(); i++) {
      if (!equal_approx(first[i], second[i], error)) return false;
    }

    return true;
  }

  constexpr size_t pair_index(size_t first, size_t second, size_t n) {
    return first * n - first * (first + 1) / 2 + second - first - 1;
  }
} // namespace neonsvm::utility

#endif
