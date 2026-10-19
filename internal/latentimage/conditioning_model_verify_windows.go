//go:build windows

package latentimage

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/graphruntime"
	"overgo/internal/hfgguf"
	"overgo/internal/hfrepo"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"path/filepath"
	"slices"
)

func verifyMOSSConditioningModel(ctx context.Context, runtime *graphruntime.ResidentSession, modelRoot, root string, output io.Writer) error {
	open := func(name, digest string) (*safetensors.Source, error) {
		path := filepath.Join(root, "conditioning-cuda", name)
		id, err := artifact.IdentifyFile(artifact.KindFile, path)
		if err != nil {
			return nil, err
		}
		if id.DigestHex() != digest {
			return nil, fmt.Errorf("MOSS conditioning fixture identity differs: %s", path)
		}
		return safetensors.OpenFile(path)
	}
	authority, err := open("conditioning-cuda.safetensors", "8f8f92966bd331e4e5568a4d7b85d4868a6d092125cbf2c54a238138af84177b")
	if err != nil {
		return err
	}
	defer authority.Close()
	embeddings, err := open("conditioning-cuda-layer0.safetensors", "839c5d451c61706e2ece1b8aab1c438c444b808527b582d6c31537ca9a7b2c33")
	if err != nil {
		return err
	}
	defer embeddings.Close()
	repo, err := hfrepo.Open(filepath.Join(modelRoot, "text_encoder"))
	if err != nil {
		return err
	}
	defer repo.Close()
	catalog, err := hfgguf.InspectLanguageModel(repo, hfgguf.LanguageHidden)
	if err != nil {
		return err
	}
	for _, name := range []string{"english.0", "english.1", "chinese.0", "chinese.1", "layout.0", "layout.1"} {
		hidden, err := safetensors.ReadF32(embeddings.Tensors[name+".embedding"])
		if err != nil {
			return err
		}
		tokens := len(hidden) / int(catalog.Spec.EmbeddingLength)
		for layer := range int(catalog.Spec.BlockCount) {
			err := func() error {
				p, err := compileModelEncoder(catalog, tokens, layer, layer+1)
				if err != nil {
					return err
				}
				names := slices.Sorted(maps.Keys(p.boundaries))
				var outputs []*tensor.Tensor
				for _, label := range names {
					outputs = append(outputs, p.boundaries[label])
				}
				program, err := p.resident(ctx, runtime, catalog, modelRoot, outputs...)
				if err != nil {
					return err
				}
				defer runtime.Release(context.WithoutCancel(ctx), program)
				p.feeds[p.input] = reference.Value{Shape: p.input.Shape, Data: hidden}
				previous := map[string][sha256.Size]byte{}
				for pass := range tensor.PairedExtent {
					actual, err := runtime.Execute(ctx, program, p.feeds)
					if err != nil {
						return err
					}
					for _, label := range names {
						want, err := safetensors.ReadF32(authority.Tensors[name+"."+label])
						if err != nil {
							return err
						}
						got := actual[p.boundaries[label]].Data
						failed, unequal, worst := compareMOSSValues(got, want, mossStageTolerance{})
						hash := sha256.Sum256(driver.Bytes(got))
						repeated := pass == 0 || hash == previous[label]
						previous[label] = hash
						if err := json.NewEncoder(output).Encode(map[string]any{"stage": "conditioning-model", "case": name, "boundary": label, "pass": pass, "values": len(want), "failed": failed, "unequal": unequal, "max_abs": worst, "repeat_checked": pass != 0, "repeat_equal": repeated}); err != nil {
							return err
						}
						if failed != 0 || unequal != 0 || !repeated {
							return fmt.Errorf("MOSS model conditioning %s %s differs (%d unequal values)", name, label, unequal)
						}
					}
					hidden = actual[p.output].Data
				}
				return nil
			}()
			if err != nil {
				return err
			}
		}
	}
	return nil
}
