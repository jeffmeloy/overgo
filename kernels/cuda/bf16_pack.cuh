// bf16_pack.cuh - the one F32 to BF16 pack, shared by the operation kernels and Strata's prompt path.

#pragma once

// BF16_NAN_BIT is the top mantissa bit of a BF16: a NaN keeps it set, since its own mantissa may lie below BF16's.
constexpr unsigned int BF16_NAN_BIT = 0x0040u;

// bf16_pack rounds to nearest even. A NaN stays a NaN: rounding would carry the GPU's default NaN 0x7FFFFFFF out of the
// exponent into minus zero, and a NaN whose mantissa lies below BF16's would truncate to an infinity, so a fault would
// read as a plausible value. The test is on the bits, so --use_fast_math cannot fold it away.
__device__ __forceinline__ unsigned short bf16_pack(float value) {
    unsigned int bits = __float_as_uint(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return (unsigned short) ((bits >> 16) | BF16_NAN_BIT);
    }
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (unsigned short) (bits >> 16);
}
