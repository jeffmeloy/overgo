// attention_row.cuh - one rule for an attention row's weight sum, shared by the operation kernels and Strata's.
//
// A row is always written. A NaN sum writes NaN, so a numeric fault reaches the output and no row keeps what the buffer
// held, and a zero sum writes 0, the guard against dividing by it. The NaN test is on the bits, so --use_fast_math cannot
// fold it away, and each form leaves a finite sum's arithmetic as its caller had it, so finite output bits do not move.

#pragma once

__device__ __forceinline__ bool attention_sum_is_nan(float sum) {
    return (__float_as_uint(sum) & 0x7fffffffu) > 0x7f800000u;
}

// attention_row_value is the row's value of the accumulator over the sum.
__device__ __forceinline__ float attention_row_value(float accumulated, float sum) {
    if (attention_sum_is_nan(sum)) {
        return sum;
    }
    return sum > 0.0f ? accumulated / sum : 0.0f;
}

// attention_row_inverse is what a row multiplies by, for a kernel that takes the inverse once.
__device__ __forceinline__ float attention_row_inverse(float sum) {
    if (attention_sum_is_nan(sum)) {
        return sum;
    }
    return sum > 0.0f ? 1.0f / sum : 0.0f;
}

// attention_row_frcp is attention_row_inverse for a kernel that rounds the inverse to nearest.
__device__ __forceinline__ float attention_row_frcp(float sum) {
    if (attention_sum_is_nan(sum)) {
        return sum;
    }
    return sum > 0.0f ? __frcp_rn(sum) : 0.0f;
}
