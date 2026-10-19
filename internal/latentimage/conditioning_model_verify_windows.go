//go:build windows

package latentimage

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
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
	"strings"
)

// mossConditioningTokens is the pinned pipeline's default max_sequence_length.
// Reconsider it with a separately captured prompt-truncation contract.
const mossConditioningTokens = 4096

func verifyMOSSConditioningModel(ctx context.Context, runtime *graphruntime.ResidentSession, modelRoot, root string, fromPrompt bool, output io.Writer) error {
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
	if fromPrompt {
		return verifyMOSSConditioningPrompts(ctx, runtime, modelRoot, authority, embeddings, output)
	}
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

func verifyMOSSConditioningPrompts(ctx context.Context, runtime *graphruntime.ResidentSession, modelRoot string, authority, embeddings *safetensors.Source, output io.Writer) error {
	c, err := loadModelConditioner(modelRoot, textTemplate{MaxTokens: mossConditioningTokens, EnableThinking: true})
	if err != nil {
		return err
	}
	defer c.source.Close()
	var prompts map[string][]string
	for _, metadata := range authority.Metadata {
		if err := json.Unmarshal([]byte(metadata["prompts"]), &prompts); err != nil {
			return err
		}
	}
	for _, name := range []string{"english.0", "english.1", "chinese.0", "chinese.1", "layout.0", "layout.1"} {
		base, branch, _ := strings.Cut(name, ".")
		entry, ok := prompts[base]
		if !ok || len(entry) != tensor.PairedExtent {
			return fmt.Errorf("MOSS native prompt %s missing", base)
		}
		prompt := entry[0]
		if branch == "1" {
			prompt = ""
		}
		text, embedding, err := c.input(prompt)
		if err != nil {
			return err
		}
		ids, err := io.ReadAll(authority.Tensors[name+".ids"].Reader())
		if err != nil {
			return err
		}
		if len(ids) != len(text.IDs)*binary.Size(int64(0)) {
			return fmt.Errorf("MOSS prompt %s token count differs", name)
		}
		for i, id := range text.IDs {
			if int64(binary.LittleEndian.Uint64(ids[i*binary.Size(int64(0)):])) != int64(id) {
				return fmt.Errorf("MOSS prompt %s token %d differs", name, i)
			}
		}
		wantEmbedding, err := safetensors.ReadF32(embeddings.Tensors[name+".embedding"])
		if err != nil {
			return err
		}
		if failed, unequal, _ := compareMOSSValues(embedding, wantEmbedding, mossStageTolerance{}); failed != 0 || unequal != 0 {
			return fmt.Errorf("MOSS prompt %s embedding lookup differs", name)
		}
		previous := map[string][sha256.Size]byte{}
		for pass := range tensor.PairedExtent + tensor.SingletonExtent {
			report := func(label string, got []float32) error {
				want, err := safetensors.ReadF32(authority.Tensors[name+"."+label])
				if err != nil {
					return err
				}
				failed, unequal, worst := compareMOSSValues(got, want, mossStageTolerance{})
				hash := sha256.Sum256(driver.Bytes(got))
				repeated := pass == 0 || previous[label] == hash
				previous[label] = hash
				if err := json.NewEncoder(output).Encode(map[string]any{"stage": "conditioning", "case": name, "boundary": label, "pass": pass, "tokens": len(text.IDs), "tokens_equal": true, "embedding_equal": true, "values": len(want), "failed": failed, "unequal": unequal, "max_abs": worst, "repeat_checked": pass != 0, "repeat_equal": repeated}); err != nil {
					return err
				}
				if failed != 0 || unequal != 0 || !repeated {
					return fmt.Errorf("MOSS prompt conditioning %s %s differs", name, label)
				}
				return nil
			}
			if pass < tensor.PairedExtent {
				_, err = c.encode(ctx, runtime, text, embedding, report)
			} else {
				var final []float32
				final, err = c.encode(ctx, runtime, text, embedding, nil)
				if err == nil {
					err = report("final_norm", final)
				}
			}
			if err != nil {
				return err
			}
		}
	}
	return nil
}
