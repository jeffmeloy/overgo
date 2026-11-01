package latentimage

import (
	"fmt"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"path/filepath"
	"testing"
)

func jointFixture(t *testing.T) *safetensors.Source {
	t.Helper()
	mossFixtureBytes(t, "joint.safetensors", "2c7092412f54441b286f0c50eecc04ec39c56ff72400d87b124cd0a107fe1236")
	s, err := safetensors.OpenFile(filepath.Join("..", "..", "fixtures", "moss", "joint.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func jointFeeds(t *testing.T, p *rotaryJointProgram, oracle *safetensors.Source, prefix string, storage dtype.Type) map[*tensor.Tensor]reference.Value {
	t.Helper()
	weights := mossGolden(t, "primitives")
	unit := make([]float32, p.unit.Shape.Dims[0])
	for i := range unit {
		unit[i] = 1
	}
	feeds := map[*tensor.Tensor]reference.Value{p.unit: {Shape: p.unit.Shape, Data: unit}}
	for _, input := range []struct {
		name string
		node *tensor.Tensor
	}{{"input", p.input}, {"rope", p.rope}, {"modulation", p.modulation}} {
		feeds[input.node] = reference.Value{Shape: input.node.Shape, Data: refinementValues(t, oracle, prefix+input.name)}
	}
	for _, node := range p.weightInputs {
		values := mossF32(t, weights, "transformer.weights."+node.Name)
		if storage == dtype.BF16 {
			dtype.RoundBF16Slice(values)
		}
		feeds[node] = reference.Value{Shape: node.Shape, Data: values}
	}
	return feeds
}

func TestMOSSRotaryJointGolden(t *testing.T) {
	oracle := jointFixture(t)
	spec := refinementSpec()
	spec.Layers = 2
	for _, length := range []int{2, 4} {
		for block := range 2 {
			t.Run(fmt.Sprintf("%d/%d", length, block), func(t *testing.T) {
				p, err := compileRotaryJointProgram(spec, 1e-6, length+6, block, dtype.F32)
				if err != nil {
					t.Fatal(err)
				}
				prefix := fmt.Sprintf("cpu-f32.%d.block.%d.", length, block)
				got, err := reference.Execute([]*tensor.Tensor{p.output}, jointFeeds(t, p, oracle, prefix, dtype.F32))
				if err != nil {
					t.Fatal(err)
				}
				compareRefinement(t, got[p.output].Data, refinementValues(t, oracle, prefix+"output"), 1.3e-6)
			})
		}
	}
}

func TestMOSSRotaryJointRefusesInvalidGeometry(t *testing.T) {
	for _, change := range []func(*TransformerSpec){func(s *TransformerSpec) { s.Layers = 0 }, func(s *TransformerSpec) { s.ModFields = 4 }, func(s *TransformerSpec) { s.Heads = 0 }} {
		s := refinementSpec()
		s.Layers = 2
		change(&s)
		if _, err := compileRotaryJointProgram(s, 1e-6, 8, 0, dtype.F32); err == nil {
			t.Fatal("invalid joint geometry accepted")
		}
	}
}
