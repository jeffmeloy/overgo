package latentimage

import (
	"encoding/json"
	"maps"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/graphruntime"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func decoderFixture(t *testing.T) (*safetensors.Source, spatialVAEConfig, map[string][]float32) {
	t.Helper()
	mossFixtureBytes(t, "decoder.safetensors", "e660997ebb8b0d2eff6b4116fff8eeba8b7655be8c22aed144087ad2c3d87ae5")
	s, e := safetensors.OpenFile(filepath.Join("..", "..", "fixtures", "moss", "decoder.safetensors"))
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() { s.Close() })
	var cfg spatialVAEConfig
	for _, m := range s.Metadata {
		if m["source_sha256"] != mossReferenceSource {
			t.Fatal("decoder source identity differs")
		}
		if e = json.Unmarshal([]byte(m["config"]), &cfg); e != nil {
			t.Fatal(e)
		}
	}
	// The frozen producer constructs AutoencoderKLFlux2 directly.
	cfg.ClassName = "AutoencoderKLFlux2"
	weights := map[string][]float32{}
	for n := range s.Tensors {
		if after, ok := strings.CutPrefix(n, "weights."); ok {
			weights[after] = refinementValues(t, s, n)
		}
	}
	return s, cfg, weights
}

func decoderGolden(t *testing.T, backend string, run graphruntime.Runner) {
	t.Helper()
	s, cfg, weights := decoderFixture(t)
	d, e := buildSpatialVAEDecoder(cfg, weights)
	if e != nil {
		t.Fatal(e)
	}
	for _, label := range []string{"2x2", "2x3", "3x2"} {
		t.Run(label, func(t *testing.T) {
			prefix := backend + "." + label + "."
			shape := s.Tensors[prefix+"input"].Shape
			p, e := CompileVAEProgram(d, int(shape[2]), int(shape[3]), dtype.F32)
			if e != nil {
				t.Fatal(e)
			}
			feeds := map[*tensor.Tensor]reference.Value{p.Latent: {Shape: p.Latent.Shape, Data: refinementValues(t, s, prefix+"input")}}
			for _, f := range p.feeds {
				feeds[f.node] = reference.Value{Shape: f.node.Shape, Data: f.data}
			}
			named := map[string]*tensor.Tensor{"decoded": p.decoded, "pixels": p.pixels}
			for n, node := range p.boundaries {
				if _, ok := s.Tensors[prefix+n]; ok {
					named[n] = node
				}
			}
			seen := map[*tensor.Tensor]bool{}
			var outputs []*tensor.Tensor
			for _, node := range named {
				if !seen[node] {
					outputs = append(outputs, node)
					seen[node] = true
				}
			}
			got, e := run(outputs, feeds)
			if e != nil {
				t.Fatal(e)
			}
			repeat, e := run(outputs, feeds)
			if e != nil {
				t.Fatal(e)
			}
			for name, node := range named {
				t.Run(name, func(t *testing.T) {
					hwc := got[node].Data
					c := int(node.Shape.Dims[0])
					plane := len(hwc) / c
					planar := make([]float32, len(hwc))
					for i := range plane {
						for ch := range c {
							planar[ch*plane+i] = hwc[i*c+ch]
						}
					}
					compareRefinement(t, planar, refinementValues(t, s, prefix+name), 1.3e-6)
					for i, v := range hwc {
						if math.Float32bits(v) != math.Float32bits(repeat[node].Data[i]) {
							t.Fatal("decoder boundary does not repeat exactly")
						}
					}
				})
			}
		})
	}
}

func TestMOSSDecoderGolden(t *testing.T) { decoderGolden(t, "cpu-f32", reference.Execute) }

func TestMOSSDecoderRejectsInvalidWeights(t *testing.T) {
	_, cfg, weights := decoderFixture(t)
	for _, name := range []string{"decoder.conv_in.weight", "decoder.mid_block.attentions.0.group_norm.bias", "bn.running_mean"} {
		t.Run(name, func(t *testing.T) {
			w := maps.Clone(weights)
			delete(w, name)
			if _, e := buildSpatialVAEDecoder(cfg, w); e == nil {
				t.Fatal("missing decoder weight accepted")
			}
		})
	}
	for _, variant := range []string{"nan", "negative-variance", "unexpected", "truncated"} {
		t.Run(variant, func(t *testing.T) {
			w := maps.Clone(weights)
			switch variant {
			case "nan":
				w["bn.running_mean"] = append([]float32(nil), w["bn.running_mean"]...)
				w["bn.running_mean"][0] = float32(math.NaN())
			case "negative-variance":
				w["bn.running_var"] = append([]float32(nil), w["bn.running_var"]...)
				w["bn.running_var"][0] = -1
			case "unexpected":
				w["decoder.extra"] = []float32{1}
			case "truncated":
				w["decoder.conv_out.weight"] = w["decoder.conv_out.weight"][:1]
			}
			if _, e := buildSpatialVAEDecoder(cfg, w); e == nil {
				t.Fatal("invalid decoder weights accepted")
			}
		})
	}
	for _, variant := range []string{"groups", "patch", "block", "layers"} {
		t.Run(variant, func(t *testing.T) {
			c := cfg
			switch variant {
			case "groups":
				c.Groups = 3
			case "patch":
				c.Patch = []int{0, 2}
			case "block":
				c.Blocks = []string{"unsupported"}
			case "layers":
				c.Layers = 0
			}
			if _, e := buildSpatialVAEDecoder(c, weights); e == nil {
				t.Fatal("invalid decoder configuration accepted")
			}
		})
	}
}
