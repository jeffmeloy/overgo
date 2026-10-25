package executor

import (
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// TestPromptAttentionKeepsTheF32PathByDefault holds a prompt's causal attention, admissible to the tensor-core kernel
// in every other way (a kernel width, grouped heads, more queries than the strided-batched path's floor), off that
// kernel unless its model admits it: the kernel's half-precision products moved dense models' held-out quality
// (gemma-4-12B-it-fp8's BBH 0.657 to 0.643, restored to 0.657 by the f32 path at the same commit, 2026-10-24), which
// no speed path may.
func TestPromptAttentionKeepsTheF32PathByDefault(t *testing.T) {
	t.Parallel()
	for _, admitted := range []bool{false, true} {
		builder := tensor.NewBuilder()
		shape := tensor.MustShape(128, 4, 96)
		query := builder.Input("query", dtype.F32, shape)
		key := builder.Input("key", dtype.F32, shape)
		value := builder.Input("value", dtype.F32, shape)
		attention := builder.AttentionWithOptions(query, key, value, tensor.AttentionOptions{Scale: 0.0883883, Causal: true, TensorCores: admitted})
		if err := builder.Err(); err != nil {
			t.Fatal(err)
		}
		attributes, ok := attention.Attrs.(tensor.AttentionAttributes)
		if !ok {
			t.Fatalf("the attention's attributes are %T", attention.Attrs)
		}
		if flashAttention(attention, attributes) != admitted {
			t.Fatalf("a causal prompt's attention reads on tensor cores: %t, want %t, as its model admits", !admitted, admitted)
		}
	}
}
