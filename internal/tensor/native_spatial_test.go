package tensor

import (
	"testing"

	"overgo/internal/tensor/dtype"
)

func TestNativeSpatialPoliciesRejectUnsupportedInputs(t *testing.T) {
	for _, test := range []struct {
		name    string
		width   uint64
		kind    dtype.Type
		options AttentionOptions
	}{
		{"narrow", 128, dtype.F32, AttentionOptions{}},
		{"bf16", 384, dtype.BF16, AttentionOptions{}},
		{"causal", 384, dtype.F32, AttentionOptions{Causal: true}},
		{"naive", 384, dtype.F32, AttentionOptions{NaiveF32: true}},
		{"window", 384, dtype.F32, AttentionOptions{Window: 2}},
		{"softcap", 384, dtype.F32, AttentionOptions{Softcap: 2}},
	} {
		t.Run(test.name, func(t *testing.T) {
			b := NewBuilder()
			x := b.Input("x", test.kind, MustShape(test.width, 1, 4))
			opts := test.options
			opts.MemoryEfficientF32 = true
			opts.Scale = 1
			if out := b.AttentionWithOptions(x, x, x, opts); out != nil || b.Err() == nil {
				t.Fatal("unsupported attention policy accepted")
			}
		})
	}
	for _, test := range []struct {
		name   string
		kernel uint64
		kind   dtype.Type
	}{{"pointwise", 1, dtype.F32}, {"five", 5, dtype.F32}, {"bf16", 3, dtype.BF16}} {
		t.Run(test.name, func(t *testing.T) {
			b := NewBuilder()
			x := b.Input("x", test.kind, MustShape(8, 7, 5))
			w := b.Input("w", test.kind, MustShape(test.kernel, test.kernel, 8, 8))
			if out := b.Conv2DWinogradF32(x, w, nil); out != nil || b.Err() == nil {
				t.Fatal("unsupported Winograd policy accepted")
			}
		})
	}
}
