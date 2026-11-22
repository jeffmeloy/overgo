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

// ArithmeticComfyUIF32 computes the native conditioning graph as ComfyUI computes a text encoder: its activations in F32
// (sd1_clip asks the embeddings and the transformer for torch.float32), each product exact over weights read in F32,
// no rounding to BF16 between operations, and attention and rotary in F32. On the MOSS Ultra encoder Transformers' BF16
// arithmetic parted from ComfyUI's states by 0.3% a layer, 0.013 to 0.021 in relative L2 at the conditioning, where
// this reads 0.00004 at most (moss-encoder-own). Admission is that of ArithmeticTransformersBF16.
const ArithmeticComfyUIF32 ArithmeticPolicy = "comfyui-f32"

func (s Spec) validateArithmetic(profile ArchitectureProfile) error {
	if s.Arithmetic == "" {
		return nil
	}
	if !s.Arithmetic.native() || profile.Attention != AttentionGatedDelta || profile.AttentionGraph.GatedDelta != gatedDeltaSeparateProjections || s.ExpertCount != 0 || s.NextNPredictLayers != 0 || s.HyperConnectionCount != 0 {
		return fmt.Errorf("source arithmetic %q requires dense separate-projection GDN conditioning without prediction layers", s.Arithmetic)
	}
	return nil
}

// native says the policy computes the source's own conditioning graph, its gated-delta scan in chunks and its native
// operations, whatever precision it rounds its activations to.
func (p ArithmeticPolicy) native() bool {
	return p == ArithmeticTransformersBF16 || p == ArithmeticComfyUIF32
}

func (p ArithmeticPolicy) round(b *tensor.Builder, x *tensor.Tensor) *tensor.Tensor {
	if p == ArithmeticTransformersBF16 {
		return b.BF16RoundNative(x)
	}
	return x
}

func (p ArithmeticPolicy) silu(b *tensor.Builder, x *tensor.Tensor) *tensor.Tensor {
	if p.native() {
		return b.SiLUNative(x)
	}
	return b.SiLU(x)
}

func (p ArithmeticPolicy) project(b *tensor.Builder, w, x *tensor.Tensor) *tensor.Tensor {
	if p == ArithmeticTransformersBF16 {
		return b.BF16RoundNative(b.MulMatWithCompute(w, x, tensor.MulMatComputeNativeTensorCore))
	}
	return b.MulMat(w, x)
}

func (p ArithmeticPolicy) weightedRMS(b *tensor.Builder, x, w *tensor.Tensor, epsilon float32) *tensor.Tensor {
	switch p {
	case ArithmeticTransformersBF16:
		return b.BF16Round(b.Multiply(b.RMSNormNative(x, epsilon), w))
	case ArithmeticComfyUIF32:
		return b.Multiply(b.RMSNormNative(x, epsilon), w)
	}
	return b.WeightedRMSNorm(x, w, epsilon)
}
