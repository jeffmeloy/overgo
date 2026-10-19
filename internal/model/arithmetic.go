package model

import (
	"fmt"
	"overgo/internal/tensor"
)

// ArithmeticPolicy binds source numerical semantics to the model specification.
type ArithmeticPolicy string

// ArithmeticTransformersBF16 preserves native Transformers BF16 conditioning.
// Admission is limited to the validated dense, separate-projection GDN topology.
const ArithmeticTransformersBF16 ArithmeticPolicy = "transformers-bf16"

func (s Spec) validateArithmetic(profile ArchitectureProfile) error {
	if s.Arithmetic == "" {
		return nil
	}
	if s.Arithmetic != ArithmeticTransformersBF16 || profile.Attention != AttentionGatedDelta || profile.AttentionGraph.GatedDelta != gatedDeltaSeparateProjections || s.ExpertCount != 0 || s.NextNPredictLayers != 0 || s.HyperConnectionCount != 0 {
		return fmt.Errorf("source arithmetic %q requires dense separate-projection GDN conditioning without prediction layers", s.Arithmetic)
	}
	return nil
}

func (p ArithmeticPolicy) round(b *tensor.Builder, x *tensor.Tensor) *tensor.Tensor {
	if p == ArithmeticTransformersBF16 {
		return b.BF16RoundNative(x)
	}
	return x
}

func (p ArithmeticPolicy) project(b *tensor.Builder, w, x *tensor.Tensor) *tensor.Tensor {
	if p == ArithmeticTransformersBF16 {
		return b.BF16RoundNative(b.MulMatWithCompute(w, x, tensor.MulMatComputeNativeTensorCore))
	}
	return b.MulMat(w, x)
}

func (p ArithmeticPolicy) weightedRMS(b *tensor.Builder, x, w *tensor.Tensor, epsilon float32) *tensor.Tensor {
	if p == ArithmeticTransformersBF16 {
		return b.BF16Round(b.Multiply(b.RMSNormNative(x, epsilon), w))
	}
	return b.WeightedRMSNorm(x, w, epsilon)
}
