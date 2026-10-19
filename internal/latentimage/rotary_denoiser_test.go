package latentimage

import (
	"fmt"
	"maps"
	"math"
	"path/filepath"
	"testing"

	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func denoiserFixture(t *testing.T) *safetensors.Source {
	t.Helper()
	mossFixtureBytes(t, "denoiser.safetensors", "924354f7ca521daaa06d85a048a91c885e2e2ddaa0bfd9b4afbdc8032be81f40")
	s, err := safetensors.OpenFile(filepath.Join("..", "..", "fixtures", "moss", "denoiser.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func denoiserSpec() TransformerSpec {
	s := refinementSpec()
	s.Layers, s.InChannels, s.TimestepEmbed, s.RopeTheta = 2, 4, 8, 10000
	return s
}

func denoiserFeeds(t *testing.T, p *rotaryDenoiserProgram, index int, storage dtype.Type) map[*tensor.Tensor]reference.Value {
	t.Helper()
	s := mossGolden(t, "primitives")
	input := mossF32(t, s, "transformer.input")
	context := mossF32(t, s, "transformer.context")
	times := mossF32(t, s, "transformer.times")
	feeds, err := p.feeds(input[index*24:(index+1)*24], context[index*32:index*32+p.text*8], times[index])
	if err != nil {
		t.Fatal(err)
	}
	for _, node := range p.weightInputs {
		values := mossF32(t, s, "transformer.weights."+node.Name)
		if storage == dtype.BF16 {
			dtype.RoundBF16Slice(values)
		}
		feeds[node] = reference.Value{Shape: node.Shape, Data: values}
	}
	return feeds
}

func denoiserOutputs(p *rotaryDenoiserProgram) ([]*tensor.Tensor, map[string]*tensor.Tensor) {
	named := map[string]*tensor.Tensor{"output": p.output}
	maps.Copy(named, p.boundaries)
	for old, key := range map[string]string{"timestep_embedding": "time", "head_modulation": "head", "text_refinement": "refinement"} {
		named[key] = named[old]
		delete(named, old)
	}
	outputs := []*tensor.Tensor{p.output}
	for _, node := range p.boundaries {
		outputs = append(outputs, node)
	}
	return outputs, named
}

func TestMOSSRotaryDenoiserGolden(t *testing.T) {
	s := denoiserFixture(t)
	for index, length := range []int{4, 2} {
		for _, variant := range []struct {
			name string
			skip []int
		}{{"all", nil}, {"skip-1", []int{1}}, {"skip-all", []int{0, 1}}} {
			t.Run(fmt.Sprintf("%d/%s", length, variant.name), func(t *testing.T) {
				p, err := compileRotaryDenoiserProgram(denoiserSpec(), 1e-6, 2, 3, length, variant.skip, dtype.F32)
				if err != nil {
					t.Fatal(err)
				}
				outputs, named := denoiserOutputs(p)
				got, err := reference.Execute(outputs, denoiserFeeds(t, p, index, dtype.F32))
				if err != nil {
					t.Fatal(err)
				}
				prefix := fmt.Sprintf("cpu-f32.%d.%s.", length, variant.name)
				for name, node := range named {
					t.Run(name, func(t *testing.T) { compareRefinement(t, got[node].Data, refinementValues(t, s, prefix+name), 1.3e-6) })
				}
			})
		}
	}
}

func TestMOSSRotaryDenoiserRefusesInvalidInput(t *testing.T) {
	for _, skip := range [][]int{{-1}, {2}} {
		if _, err := compileRotaryDenoiserProgram(denoiserSpec(), 1e-6, 2, 3, 4, skip, dtype.F32); err == nil {
			t.Fatal("invalid skip accepted")
		}
	}
	for _, change := range []func(*TransformerSpec){func(s *TransformerSpec) { s.TimestepEmbed = 3 }, func(s *TransformerSpec) { s.HeadDim = 7 }, func(s *TransformerSpec) { s.RopeTheta = math.Inf(1) }, func(s *TransformerSpec) { s.Layers = 0 }} {
		s := denoiserSpec()
		change(&s)
		if _, err := compileRotaryDenoiserProgram(s, 1e-6, 2, 3, 4, nil, dtype.F32); err == nil {
			t.Fatal("invalid geometry accepted")
		}
	}
	p, err := compileRotaryDenoiserProgram(denoiserSpec(), 1e-6, 2, 3, 4, nil, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	for _, input := range []struct {
		latent, context []float32
		time            float32
	}{{nil, make([]float32, 32), 1}, {make([]float32, 24), nil, 1}, {make([]float32, 24), make([]float32, 32), float32(math.NaN())}} {
		if _, err := p.feeds(input.latent, input.context, input.time); err == nil {
			t.Fatal("invalid feeds accepted")
		}
	}
	if _, err := coordinateRotary(8, math.MaxInt, 2, 1, 10000); err == nil {
		t.Fatal("overflow accepted")
	}
}
