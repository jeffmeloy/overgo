package latentimage

import (
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func setBuilderMatmulCompute(builder *tensor.Builder, storage dtype.Type) {
	if storage == dtype.BF16 {
		builder.SetMulMatCompute(tensor.MulMatComputeBF16TensorCore)
	}
}

func roundAttentionForStorage(
	builder *tensor.Builder,
	storage dtype.Type,
	query, key, value *tensor.Tensor,
) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
	if storage == dtype.BF16 {
		return builder.BF16Round(query), builder.BF16Round(key), builder.BF16Round(value)
	}
	return query, key, value
}
