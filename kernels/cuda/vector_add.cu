extern "C" __global__ void vector_add(
        const float * input_a,
        const float * input_b,
        float * output,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = input_a[index] + input_b[index];
    }
}
