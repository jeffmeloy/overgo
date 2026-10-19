//go:build windows

package latentimage

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"math"
	"path/filepath"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/graphruntime"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// mossRefinementBlocks and mossRefinementEpsilon belong to the pinned TextEmbedder.
// Reconsider these constants only when admitting a different source revision.
const (
	mossRefinementBlocks  = 2
	mossRefinementEpsilon = 1e-6
)

// VerifyMOSSTransformer compares a resident transformer stage with pinned native cases.
// Checkpoint and oracle documents must retain their frozen content identities.
// Every case must meet upstream BF16 tolerance and repeat bitwise.
func VerifyMOSSTransformer(ctx context.Context, modelRoot, referenceRoot, contractRoot, stage string, output io.Writer) error {
	if stage != "refinement" && stage != "joint" && stage != "denoiser" {
		return fmt.Errorf("MOSS stage %q must be refinement, joint or denoiser", stage)
	}
	verify := func(path, digest string) error {
		id, err := artifact.IdentifyFile(artifact.KindFile, path)
		if err != nil {
			return err
		}
		if id.DigestHex() != digest {
			return fmt.Errorf("MOSS reference identity differs: %s", path)
		}
		return nil
	}
	checkpointPath := filepath.Join(contractRoot, "checkpoint.json")
	contractPath := filepath.Join(contractRoot, "oracle-contract.json")
	if err := verify(checkpointPath, "0d9b600f6232957565c26e86c9f9b0d19411b87fe97d665bbe7b93bd7770a9db"); err != nil {
		return err
	}
	if err := verify(contractPath, "0c2767456a75075671de88538d8c1ce0f673ef41c2bae897c4e59449fad0ca86"); err != nil {
		return err
	}
	var checkpoint struct {
		Files []struct{ Path, SHA256 string }
	}
	if err := readJSON(checkpointPath, &checkpoint); err != nil {
		return err
	}
	for _, f := range checkpoint.Files {
		if filepath.Dir(f.Path) == "transformer" {
			if err := verify(filepath.Join(modelRoot, f.Path), f.SHA256); err != nil {
				return err
			}
		}
	}
	var config struct {
		Layers    int     `json:"num_layers"`
		Dim       int     `json:"dim"`
		FFN       int     `json:"ffn_dim"`
		Heads     int     `json:"num_heads"`
		Text      int     `json:"text_dim"`
		Channels  int     `json:"latent_channels"`
		Frequency int     `json:"freq_dim"`
		RopeBase  float64 `json:"rope_base"`
	}
	if err := readJSON(filepath.Join(modelRoot, "transformer", "config.json"), &config); err != nil {
		return err
	}
	if config.Heads <= 0 {
		return fmt.Errorf("MOSS head count must be positive")
	}
	spec := TransformerSpec{Layers: config.Layers, Hidden: config.Dim, Heads: config.Heads, HeadDim: config.Dim / config.Heads, Intermediate: config.FFN, TextHidden: config.Text, RefinerTextBlocks: mossRefinementBlocks, InChannels: config.Channels, TimestepEmbed: config.Frequency, RopeTheta: config.RopeBase}
	var contract struct {
		Cases []struct {
			Name     string
			Artifact artifact.ID
		}
		Comparison struct {
			Tolerances map[string]struct {
				Absolute float64 `json:"atol"`
				Relative float64 `json:"rtol"`
			} `json:"dtype_tolerances"`
		}
	}
	if err := readJSON(contractPath, &contract); err != nil {
		return err
	}
	jointPath := filepath.Join(referenceRoot, "joint-checkpoint.safetensors")
	if stage == "joint" {
		if err := verify(jointPath, "17497c2fffae48c580104e1cd2b9f849af064802d45777895cc38122fd793347"); err != nil {
			return err
		}
	}
	runtime, err := graphruntime.NewResidentSession(0)
	if err != nil {
		return err
	}
	defer runtime.Close(context.WithoutCancel(ctx))
	for _, c := range contract.Cases {
		if !slices.Contains([]string{"full-english", "full-chinese-rectangle", "full-layout"}, c.Name) {
			continue
		}
		path := filepath.Join(referenceRoot, c.Name, "full_reference.safetensors")
		if err := verify(path, c.Artifact.DigestHex()); err != nil {
			return err
		}
		hiddenName, ropeName, wantName, hiddenWidth := "conditioning.0.hidden", "rope", "text_refinement", spec.TextHidden
		if stage == "joint" {
			path = jointPath
			hiddenName, ropeName, wantName, hiddenWidth = c.Name+".input", c.Name+".rope", c.Name+".output", spec.Hidden
		}
		err := func() error {
			source, err := safetensors.OpenFile(path)
			if err != nil {
				return err
			}
			defer source.Close()
			hidden := source.Tensors[hiddenName]
			if len(hidden.Shape) != tensor.TripleExtent || hidden.Shape[0] != tensor.SingletonExtent || hidden.Shape[tensor.PairedExtent] != uint64(hiddenWidth) {
				return fmt.Errorf("MOSS stage input shape differs: %v", hidden.Shape)
			}
			tokens := int(hidden.Shape[1])
			inputs := map[string]*tensor.Tensor{}
			feeds := map[*tensor.Tensor]reference.Value{}
			var result, rotary *tensor.Tensor
			var weights tensor.WeightBindings
			named := map[string]*tensor.Tensor{}
			if stage == "refinement" {
				p, err := compileRotaryRefinementProgram(spec, mossRefinementEpsilon, tokens, dtype.BF16)
				if err != nil {
					return err
				}
				inputs[hiddenName], inputs[ropeName] = p.InEncoder, p.InRoPE
				result, rotary, weights = p.Fused, p.InRoPE, p.weightInputs
			} else if stage == "joint" {
				p, err := compileRotaryJointProgram(spec, mossRefinementEpsilon, tokens, 0, dtype.BF16)
				if err != nil {
					return err
				}
				inputs[hiddenName], inputs[ropeName], inputs[c.Name+".modulation"] = p.input, p.rope, p.modulation
				unit := make([]float32, spec.Hidden)
				for i := range unit {
					unit[i] = tensor.UnitScale
				}
				feeds[p.unit] = reference.Value{Shape: p.unit.Shape, Data: unit}
				result, rotary, weights = p.output, p.rope, p.weightInputs
			} else {
				latent := source.Tensors["initial_latents"]
				if len(latent.Shape) != rotaryCoordinateAxes || latent.Shape[0] != tensor.SingletonExtent || latent.Shape[1] != uint64(spec.InChannels) {
					return fmt.Errorf("MOSS latent input shape differs: %v", latent.Shape)
				}
				p, err := compileRotaryDenoiserProgram(spec, mossRefinementEpsilon, int(latent.Shape[tensor.PairedExtent]), int(latent.Shape[tensor.TripleExtent]), tokens, nil, dtype.BF16)
				if err != nil {
					return err
				}
				values, err := safetensors.ReadF32(latent)
				if err != nil {
					return err
				}
				context, err := safetensors.ReadF32(hidden)
				if err != nil {
					return err
				}
				times, err := safetensors.ReadF32(source.Tensors["schedule.timesteps"])
				if err != nil {
					return err
				}
				if len(times) == 0 {
					return fmt.Errorf("MOSS timestep schedule is empty")
				}
				feeds, err = p.feeds(values, context, times[0])
				if err != nil {
					return err
				}
				result, weights, wantName = p.output, p.weightInputs, "prediction.0"
				for name, node := range p.boundaries {
					if _, ok := source.Tensors[name]; ok {
						named[name] = node
					}
				}
			}
			named[wantName] = result
			names := slices.Sorted(maps.Keys(named))
			outputs := make([]*tensor.Tensor, 0, len(names))
			for _, name := range names {
				outputs = append(outputs, named[name])
			}
			program, err := compileResidentWeights(ctx, runtime, "MOSS "+stage, filepath.Join(modelRoot, "transformer"), weights, nil, outputs...)
			if err != nil {
				return err
			}
			defer runtime.Release(context.WithoutCancel(ctx), program)
			for name, node := range inputs {
				values, err := safetensors.ReadF32(source.Tensors[name])
				if err != nil {
					return err
				}
				if node == rotary {
					if len(values) < tokens*spec.HeadDim {
						return fmt.Errorf("MOSS rotary values are truncated")
					}
					values = values[len(values)-tokens*spec.HeadDim:]
				}
				feeds[node] = reference.Value{Shape: node.Shape, Data: values}
			}
			got, err := runtime.Execute(ctx, program, feeds)
			if err != nil {
				return err
			}
			repeat, err := runtime.Execute(ctx, program, feeds)
			if err != nil {
				return err
			}
			for _, name := range names {
				want, err := safetensors.ReadF32(source.Tensors[name])
				if err != nil {
					return err
				}
				actual, repeated := got[named[name]].Data, repeat[named[name]].Data
				if len(actual) != len(want) || len(repeated) != len(want) {
					return fmt.Errorf("MOSS %s output lengths differ", name)
				}
				failures, unequal, repeatFailures := 0, 0, 0
				var worst float64
				tolerance := contract.Comparison.Tolerances["BF16"]
				for i, x := range actual {
					difference := math.Abs(float64(x) - float64(want[i]))
					if !checked.Finite32(x) || !checked.Finite32(want[i]) || difference > tolerance.Absolute+tolerance.Relative*math.Abs(float64(want[i])) {
						failures++
					}
					if checked.Finite64(difference) {
						worst = max(worst, difference)
					}
					if math.Float32bits(x) != math.Float32bits(want[i]) {
						unequal++
					}
					if math.Float32bits(x) != math.Float32bits(repeated[i]) {
						repeatFailures++
					}
				}
				if err := json.NewEncoder(output).Encode(map[string]any{"stage": stage, "boundary": name, "case": c.Name, "reference": c.Artifact, "values": len(want), "failed": failures, "unequal": unequal, "repeat_failed": repeatFailures, "max_abs": worst, "resident_weight_bytes": runtime.StaticBytes()}); err != nil {
					return err
				}
				if failures != 0 || repeatFailures != 0 {
					return fmt.Errorf("MOSS %s violates frozen comparison in %s", stage, c.Name)
				}
			}
			return nil
		}()
		if err != nil {
			return err
		}
	}
	return nil
}
