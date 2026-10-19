//go:build windows

package latentimage

import (
	"fmt"
	"math"
	"path/filepath"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestMOSSDecoderCUDAGolden(t *testing.T) {
	cudatest.Require(t)
	run, _, close := cudaVAERunner(t)
	defer close()
	decoderGolden(t, "cuda-f32", run)
}

func TestMOSSDecoderSiLU(t *testing.T) {
	cudatest.Require(t)
	mossFixtureBytes(t, "silu.safetensors", "b5b6ab47e5878e4569e1b340fbbf515745f70323183b0b16e8883cd02c536006")
	source, err := safetensors.OpenFile(filepath.Join("..", "..", "fixtures", "moss", "silu.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	run, _, close := cudaVAERunner(t)
	defer close()
	values := refinementValues(t, source, "input")
	b := tensor.NewBuilder()
	x := b.Input("x", dtype.F32, tensor.MustShape(uint64(len(values))))
	out := b.SiLU(x)
	got, err := run([]*tensor.Tensor{out}, map[*tensor.Tensor]reference.Value{x: {Shape: x.Shape, Data: values}})
	if err != nil {
		t.Fatal(err)
	}
	want := refinementValues(t, source, "output")
	for i, value := range got[out].Data {
		if math.Float32bits(value) != math.Float32bits(want[i]) {
			t.Fatalf("native SiLU differs at %d: got=%g want=%g", i, value, want[i])
		}
	}
}

func TestMOSSDecoderAttention(t *testing.T) {
	cudatest.Require(t)
	mossFixtureBytes(t, "attention-f32.safetensors", "9569524afb160ea282781917920afce959686dd2f8970104633882f76d381164")
	source, err := safetensors.OpenFile(filepath.Join("..", "..", "fixtures", "moss", "attention-f32.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	run, _, close := cudaVAERunner(t)
	defer close()
	for _, shape := range [][3]uint64{{144, 17, 65}, {256, 32, 128}, {384, 32, 128}, {384, 33, 257}, {384, 32, 384}} {
		for _, mode := range []string{"random", "uniform", "basis"} {
			c, n, m := shape[0], shape[1], shape[2]
			name := fmt.Sprintf("%d-%d-%d-%s", c, n, m, mode)
			t.Run(name, func(t *testing.T) {
				b := tensor.NewBuilder()
				q := b.Input("q", dtype.F32, tensor.MustShape(c, 1, n))
				k := b.Input("k", dtype.F32, tensor.MustShape(c, 1, m))
				v := b.Input("v", dtype.F32, k.Shape)
				out := b.AttentionWithOptions(q, k, v, tensor.AttentionOptions{Scale: float32(1 / math.Sqrt(float64(c))), MemoryEfficientF32: true})
				feeds := map[*tensor.Tensor]reference.Value{}
				for key, node := range map[string]*tensor.Tensor{"q": q, "k": k, "v": v} {
					feeds[node] = reference.Value{Shape: node.Shape, Data: refinementValues(t, source, name+"."+key)}
				}
				got, e := run([]*tensor.Tensor{out}, feeds)
				if e != nil {
					t.Fatal(e)
				}
				again, e := run([]*tensor.Tensor{out}, feeds)
				if e != nil {
					t.Fatal(e)
				}
				want := refinementValues(t, source, name+".out")
				actual := got[out].Data
				failed, unequal, worst := compareMOSSValues(actual, want, mossStageTolerance{Absolute: 1e-5, Relative: 1.3e-6})
				for i, value := range actual {
					if math.Float32bits(value) != math.Float32bits(again[out].Data[i]) {
						t.Fatal("attention does not repeat exactly")
					}
				}
				t.Logf("values=%d unequal=%d max_abs=%g", len(want), unequal, worst)
				if failed != 0 || c%32 == 0 && unequal != 0 {
					t.Fatalf("native attention differs: failed=%d unequal=%d", failed, unequal)
				}
			})
		}
	}
}

func TestMOSSDecoderGroupNormalization(t *testing.T) {
	cudatest.Require(t)
	mossFixtureBytes(t, "group-norm.safetensors", "8400b887c0ae03db7ed9585d2b6a153594e2920dc0efd548c621886263727511")
	source, err := safetensors.OpenFile(filepath.Join("..", "..", "fixtures", "moss", "group-norm.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	run, _, close := cudaVAERunner(t)
	defer close()
	for _, shape := range [][3]int{{8, 16, 4}, {16, 17, 4}, {32, 1, 4}, {96, 512, 32}, {384, 2048, 32}} {
		c, n, g := shape[0], shape[1], shape[2]
		name := fmt.Sprintf("%d-%d-%d", c, n, g)
		t.Run(name, func(t *testing.T) {
			b := tensor.NewBuilder()
			input := b.Input("input", dtype.F32, tensor.MustShape(uint64(c), uint64(n)))
			weight := b.Input("weight", dtype.F32, tensor.MustShape(uint64(c)))
			bias := b.Input("bias", dtype.F32, weight.Shape)
			out := b.GroupNorm(input, weight, bias, uint32(g), 1e-6)
			planar := refinementValues(t, source, name+".input")
			hwc := make([]float32, len(planar))
			for p := range n {
				for ch := range c {
					hwc[p*c+ch] = planar[ch*n+p]
				}
			}
			feeds := map[*tensor.Tensor]reference.Value{input: {Shape: input.Shape, Data: hwc}, weight: {Shape: weight.Shape, Data: refinementValues(t, source, name+".weight")}, bias: {Shape: bias.Shape, Data: refinementValues(t, source, name+".bias")}}
			got, e := run([]*tensor.Tensor{out}, feeds)
			if e != nil {
				t.Fatal(e)
			}
			again, e := run([]*tensor.Tensor{out}, feeds)
			if e != nil {
				t.Fatal(e)
			}
			want := refinementValues(t, source, name+".cuda")
			actual := got[out].Data
			unequal := 0
			for p := range n {
				for ch := range c {
					index := p*c + ch
					if math.Float32bits(actual[index]) != math.Float32bits(want[ch*n+p]) {
						unequal++
					}
					if math.Float32bits(actual[index]) != math.Float32bits(again[out].Data[index]) {
						t.Fatal("group normalization does not repeat exactly")
					}
				}
			}
			t.Logf("values=%d unequal=%d", len(actual), unequal)
			if unequal > 0 {
				t.Errorf("native CUDA group normalization differs in %d values", unequal)
			}
		})
	}
}
