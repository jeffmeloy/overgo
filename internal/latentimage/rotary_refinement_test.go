package latentimage

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"path/filepath"
	"testing"
)

func refinementFixture(t *testing.T) *safetensors.Source {
	t.Helper()
	mossFixtureBytes(t, "refinement.safetensors", "91d10d4247ccdf97efb903eaee2bd2ceb1ede020d34a9622eec4c0350d7850d7")
	s, e := safetensors.OpenFile(filepath.Join("..", "..", "fixtures", "moss", "refinement.safetensors"))
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() { s.Close() })
	for _, m := range s.Metadata {
		if m["source_sha256"] != mossReferenceSource || m["authority_sha256"] != "f55669402c6d12d20d047a461e90fa9e2da3f8bdb7b3d55cc189cc85f84b2160" {
			t.Fatal("refinement provenance differs")
		}
	}
	return s
}
func refinementValues(t *testing.T, s *safetensors.Source, name string) []float32 {
	t.Helper()
	v, ok := s.Tensors[name]
	if !ok {
		t.Fatal(name)
	}
	r, e := safetensors.F32Reader(v)
	if e != nil {
		t.Fatal(e)
	}
	b, e := io.ReadAll(r)
	if e != nil {
		t.Fatal(e)
	}
	x := make([]float32, len(b)/4)
	for i := range x {
		x[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return x
}
func refinementSpec() TransformerSpec {
	return TransformerSpec{Hidden: 16, Heads: 2, HeadDim: 8, Intermediate: 32, TextHidden: 8, RefinerTextBlocks: 2}
}
func refinementFeeds(t *testing.T, p *FusionProgram, s *safetensors.Source, prefix string, storage dtype.Type) map[*tensor.Tensor]reference.Value {
	t.Helper()
	weights := mossGolden(t, "primitives")
	feeds := map[*tensor.Tensor]reference.Value{}
	for _, n := range p.weightInputs {
		x := mossF32(t, weights, "transformer.weights."+n.Name)
		if storage == dtype.BF16 {
			dtype.RoundBF16Slice(x)
		}
		feeds[n] = reference.Value{Shape: n.Shape, Data: x}
	}
	feeds[p.InEncoder] = reference.Value{Shape: p.InEncoder.Shape, Data: refinementValues(t, s, prefix+"context")}
	feeds[p.InRoPE] = reference.Value{Shape: p.InRoPE.Shape, Data: refinementValues(t, s, prefix+"rope")}
	return feeds
}
func compareRefinement(t *testing.T, actual, want []float32, relative float64) {
	t.Helper()
	if len(actual) != len(want) {
		t.Fatal("refinement output geometry differs")
	}
	failed := 0
	maximum := 0.0
	for i, x := range actual {
		d := math.Abs(float64(x) - float64(want[i]))
		maximum = max(maximum, d)
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || d > 1e-5+relative*math.Abs(float64(want[i])) {
			failed++
		}
	}
	t.Logf("values=%d failed=%d max_abs=%g", len(want), failed, maximum)
	if failed != 0 {
		t.Errorf("refinement violates frozen tolerance in %d values", failed)
	}
}
func TestMOSSRotaryRefinementGolden(t *testing.T) {
	s := refinementFixture(t)
	for _, length := range []int{2, 4} {
		t.Run(fmt.Sprint(length), func(t *testing.T) {
			p, e := compileRotaryRefinementProgram(refinementSpec(), 1e-6, length, dtype.F32)
			if e != nil {
				t.Fatal(e)
			}
			prefix := fmt.Sprintf("cpu-f32.%d.", length)
			outputs := p.RefinementOutputs
			got, e := reference.Execute(outputs, refinementFeeds(t, p, s, prefix, dtype.F32))
			if e != nil {
				t.Fatal(e)
			}
			for i, n := range p.RefinementOutputs {
				compareRefinement(t, got[n].Data, refinementValues(t, s, fmt.Sprintf("%sblock.%d", prefix, i)), 1.3e-6)
			}
			compareRefinement(t, got[p.Fused].Data, refinementValues(t, s, prefix+"output"), 1.3e-6)
		})
	}
}
func TestMOSSRotaryRefinementRefusesInvalidGeometry(t *testing.T) {
	for _, change := range []func(*TransformerSpec){func(s *TransformerSpec) { s.Heads = 0 }, func(s *TransformerSpec) { s.HeadDim = 7 }, func(s *TransformerSpec) { s.Hidden = 15 }, func(s *TransformerSpec) { s.Intermediate = 0 }, func(s *TransformerSpec) { s.TextHidden = -1 }, func(s *TransformerSpec) { s.RefinerTextBlocks = 0 }} {
		s := refinementSpec()
		change(&s)
		if _, e := compileRotaryRefinementProgram(s, 1e-6, 2, dtype.F32); e == nil {
			t.Fatal("invalid geometry accepted")
		}
	}
	for _, epsilon := range []float32{0, -1, float32(math.Inf(1)), float32(math.NaN())} {
		if _, e := compileRotaryRefinementProgram(refinementSpec(), epsilon, 2, dtype.F32); e == nil {
			t.Fatal("invalid epsilon accepted")
		}
	}
	if _, e := compileRotaryRefinementProgram(refinementSpec(), 1e-6, 0, dtype.F32); e == nil {
		t.Fatal("empty text accepted")
	}
	if _, e := compileRotaryRefinementProgram(refinementSpec(), 1e-6, 2, dtype.F16); e == nil {
		t.Fatal("undeclared precision accepted")
	}
}
