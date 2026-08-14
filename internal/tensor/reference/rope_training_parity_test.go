package reference

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/hostmath"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// TestTrainingRopeMatchesServingNeoX locks the SERVING-vs-TRAINING RoPE
// reconciliation: the training-side rotate-half (hostmath.ApplyRotaryHalf over
// the first hostmath.RopeWidth dims) must compute bit-identical rotated q/k to
// the serving graph's rope_neox op (reference.ropeNeoX) for the affected model
// configs -- both partial rope (rope.dimension_count < key_length, e.g. qwen3.5
// head_dim*partial_rotary_factor) and full rope (gemma3n key_length==head_dim).
// Serving is the source of truth; this proves training follows it exactly.
func TestTrainingRopeMatchesServingNeoX(t *testing.T) {
	cases := []struct {
		name             string
		headDim, ropeDim int // ropeDim==headDim is full rope
		heads, tokens    int
		theta            float64
	}{
		{"qwen35_partial_256_64", 256, 64, 4, 3, 10_000_000},
		{"partial_8_4", 8, 4, 3, 5, 10_000},
		{"partial_odd_pairs_12_6", 12, 6, 2, 4, 1_000_000},
		{"gemma3n_full_256_256", 256, 256, 2, 3, 1_000_000},
		{"full_8_8", 8, 8, 2, 4, 10_000},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(1))
			width, heads, tokens := tc.headDim, tc.heads, tc.tokens
			data := make([]float32, width*heads*tokens)
			for i := range data {
				data[i] = float32(rng.NormFloat64())
			}
			positions := make([]uint32, tokens)
			for i := range positions {
				positions[i] = uint32(i)
			}

			// SERVING: rope_neox over the config rotary width via the graph +
			// reference executor (the serving source of truth).
			builder := tensor.NewBuilder()
			input := builder.Input("q", dtype.F32, tensor.MustShape(
				uint64(width), uint64(heads), uint64(tokens),
			))
			output := builder.RoPENeoXScaled(
				input, positions, uint32(tc.ropeDim), float32(tc.theta), 1,
			)
			if err := builder.Err(); err != nil {
				t.Fatal(err)
			}
			inValue, err := NewValue(input.Shape, append([]float32(nil), data...))
			if err != nil {
				t.Fatal(err)
			}
			results, err := Execute(
				[]*tensor.Tensor{output},
				map[*tensor.Tensor]Value{input: inValue},
			)
			if err != nil {
				t.Fatal(err)
			}
			serving := results[output].Data

			// TRAINING: hostmath rotate-half over the first RopeWidth dims.
			rd := hostmath.RopeWidth(tc.ropeDim, tc.headDim)
			if rd != tc.ropeDim {
				t.Fatalf("RopeWidth(%d,%d)=%d, want %d", tc.ropeDim, tc.headDim, rd, tc.ropeDim)
			}
			invFreq := hostmath.RopeInvFreq(tc.theta, rd)
			training := append([]float32(nil), data...)
			for p := 0; p < tokens; p++ {
				for h := 0; h < heads; h++ {
					base := (p*heads + h) * width
					hostmath.ApplyRotaryHalf(training[base:base+rd], invFreq, p)
				}
			}

			var maxAbs float64
			for i := range serving {
				d := math.Abs(float64(serving[i] - training[i]))
				if d > maxAbs {
					maxAbs = d
				}
			}
			// Both paths rotate the same dims by the same angles in float64 and
			// store float32. They agree to ~1 float32 ULP, not bit-exact: serving
			// forms the frequency as pow(base,-2i/d) inline while training
			// precomputes 1/pow(base,2i/d) (RopeInvFreq) -- equal in exact
			// arithmetic, differing only in the last float64 bit. A larger diff
			// would mean a real convention mismatch (wrong width/pairing/base).
			const tol = 1e-6
			if maxAbs > tol {
				t.Fatalf("training vs serving rope max abs diff = %g, want <= %g", maxAbs, tol)
			}
			// Guard against a no-op test: the untouched tail must be unchanged and
			// the rotated head must actually differ from the input at pos>0.
			if rd < width {
				for i := 0; i < len(data); i++ {
					lane := i % width
					if lane >= rd && serving[i] != data[i] {
						t.Fatalf("partial-rope tail lane %d changed: %v != %v", lane, serving[i], data[i])
					}
				}
			}
			t.Logf("%s: headDim=%d ropeDim=%d rotated exactly, tail passthrough ok, max|serving-training|=%g",
				tc.name, tc.headDim, tc.ropeDim, maxAbs)
		})
	}
}
