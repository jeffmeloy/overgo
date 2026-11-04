package inference

import (
	"overgo/internal/cuda/executor"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// cacheTypeOf is the type a runner's device cache keeps its keys and values at: BF16, half the bytes of F32, where
// every read of them takes a kernel that widens BF16 as it reads (executor.BF16CacheAttention), else F32. BF16 holds
// where the runner keeps its own device cache without cells or a Strata round's drafter, and every layer is an
// attention layer of its own keys and values, without other token states, whose prompt the tensor-core kernel takes:
// causal, without sinks, ALiBi, block masks or chunked windows, its key and value heads as wide as the kernel's. Its
// keys and values rounded to BF16, Qwen2.5-0.5B and MiniCPM5-1B answered BBH and IFEval without a lean against them.
// F32 keeps the F32 cache, the path a held-out check holds the BF16 cache to.
func (r *Runner) cacheTypeOf(f32 bool) dtype.Type {
	if f32 || !r.persistentDeviceCache() || r.qsaLayers != nil || r.cellLayers != nil || r.strataDraft != nil ||
		r.spec.KeyLength != r.spec.ValueLength || !executor.BF16CacheAttention(uint64(r.spec.KeyLength)) {
		return dtype.F32
	}
	for layer := range r.weights.Layers {
		plan := r.layerProgram(layer).Layer()
		attention := plan.AttentionGraph
		if plan.Recurrent || plan.SharedKV || !attention.TensorCores || !attention.Causal || attention.UseSinks ||
			attention.ChunkedWindow || attention.MaxALiBiBias != 0 || attention.BlockSize != 0 {
			return dtype.F32
		}
		schema, err := r.program.Model.CacheSchema(uint32(layer), tensor.SingletonExtent)
		if err != nil || !schema.Primary.Key.Mode.TokenAligned() || !schema.Primary.Value.Mode.TokenAligned() {
			return dtype.F32
		}
		states := false
		schema.RangeStates(func(model.CacheStateName, model.CacheState[model.CacheValueSchema]) { states = true })
		if states {
			return dtype.F32
		}
	}
	return dtype.BF16
}
