//go:build windows

package latentimage

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
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

// VerifyMOSSRefinement compares resident refinement with the pinned native cases.
// Checkpoint and oracle documents must retain their frozen content identities.
// Every case must meet upstream BF16 tolerance and repeat bitwise.
func VerifyMOSSRefinement(ctx context.Context, modelRoot, referenceRoot, contractRoot string, output io.Writer) error {
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
		Dim   int `json:"dim"`
		FFN   int `json:"ffn_dim"`
		Heads int `json:"num_heads"`
		Text  int `json:"text_dim"`
	}
	if err := readJSON(filepath.Join(modelRoot, "transformer", "config.json"), &config); err != nil {
		return err
	}
	spec := TransformerSpec{Hidden: config.Dim, Heads: config.Heads, HeadDim: config.Dim / config.Heads, Intermediate: config.FFN, TextHidden: config.Text, RefinerTextBlocks: mossRefinementBlocks}
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
		err := func() error {
			source, err := safetensors.OpenFile(path)
			if err != nil {
				return err
			}
			defer source.Close()
			hidden := source.Tensors["conditioning.0.hidden"]
			if len(hidden.Shape) != tensor.TripleExtent || hidden.Shape[0] != tensor.SingletonExtent || hidden.Shape[tensor.PairedExtent] != uint64(spec.TextHidden) {
				return fmt.Errorf("MOSS conditioning shape differs: %v", hidden.Shape)
			}
			tokens := int(hidden.Shape[1])
			p, err := compileRotaryRefinementProgram(spec, mossRefinementEpsilon, tokens, dtype.BF16)
			if err != nil {
				return err
			}
			program, err := compileResidentWeights(ctx, runtime, "MOSS rotary refinement", filepath.Join(modelRoot, "transformer"), p.weightInputs, nil, p.Fused)
			if err != nil {
				return err
			}
			defer runtime.Release(context.WithoutCancel(ctx), program)
			feeds := map[*tensor.Tensor]reference.Value{}
			for _, input := range []struct {
				name string
				node *tensor.Tensor
			}{{"conditioning.0.hidden", p.InEncoder}, {"rope", p.InRoPE}} {
				values, err := safetensors.ReadF32(source.Tensors[input.name])
				if err != nil {
					return err
				}
				if input.node == p.InRoPE {
					if len(values) < tokens*spec.HeadDim {
						return fmt.Errorf("MOSS rotary values are truncated")
					}
					values = values[len(values)-tokens*spec.HeadDim:]
				}
				feeds[input.node] = reference.Value{Shape: input.node.Shape, Data: values}
			}
			got, err := runtime.Execute(ctx, program, feeds)
			if err != nil {
				return err
			}
			repeat, err := runtime.Execute(ctx, program, feeds)
			if err != nil {
				return err
			}
			want, err := safetensors.ReadF32(source.Tensors["text_refinement"])
			if err != nil {
				return err
			}
			actual, repeated := got[p.Fused].Data, repeat[p.Fused].Data
			if len(actual) != len(want) || len(repeated) != len(want) {
				return fmt.Errorf("MOSS refinement output lengths differ")
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
			if err := json.NewEncoder(output).Encode(map[string]any{"case": c.Name, "reference": c.Artifact, "values": len(want), "failed": failures, "unequal": unequal, "repeat_failed": repeatFailures, "max_abs": worst, "resident_weight_bytes": runtime.StaticBytes()}); err != nil {
				return err
			}
			if failures != 0 || repeatFailures != 0 {
				return fmt.Errorf("MOSS refinement violates frozen comparison in %s", c.Name)
			}
			return nil
		}()
		if err != nil {
			return err
		}
	}
	return nil
}
