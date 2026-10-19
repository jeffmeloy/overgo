//go:build windows

package latentimage

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/graphruntime"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func verifyMOSSConditioningDelta(ctx context.Context, runtime *graphruntime.ResidentSession, root string, output io.Writer) error {
	var sources []*safetensors.Source
	var identities []artifact.ID
	for _, fixture := range []struct{ file, digest string }{
		{"conditioning-cuda-layer0.safetensors", "839c5d451c61706e2ece1b8aab1c438c444b808527b582d6c31537ca9a7b2c33"},
		{"conditioning-cuda-chunk.safetensors", "8257739bee1f645eb00d3dfb8a58858bd728e75e68c14c41d34fe2460100b3f7"},
	} {
		path := filepath.Join(root, "conditioning-cuda", fixture.file)
		id, err := artifact.IdentifyFile(artifact.KindFile, path)
		if err != nil {
			return err
		}
		if id.DigestHex() != fixture.digest {
			return fmt.Errorf("MOSS chunk reference identity differs: %s", path)
		}
		source, err := safetensors.OpenFile(path)
		if err != nil {
			return err
		}
		defer source.Close()
		sources = append(sources, source)
		identities = append(identities, id)
	}
	for _, name := range []string{"english.0", "english.1", "chinese.0", "chinese.1", "layout.0", "layout.1"} {
		err := func() error {
			b := tensor.NewBuilder()
			feeds := map[*tensor.Tensor]reference.Value{}
			var nodes []*tensor.Tensor
			for _, label := range []string{"l2.query", "l2.key", "delta.value", "delta.g", "delta.beta"} {
				item, ok := sources[0].Tensors[name+"."+label]
				if !ok || len(item.Shape) < tensor.TripleExtent {
					return fmt.Errorf("MOSS chunk input %s has invalid shape", label)
				}
				dims := []uint64{1, item.Shape[tensor.PairedExtent], item.Shape[1], item.Shape[0]}
				if len(item.Shape) == tensor.MaxDimensions {
					dims[0] = item.Shape[tensor.TripleExtent]
				}
				node := b.Input(label, dtype.F32, tensor.MustShape(dims...))
				values, err := safetensors.ReadF32(item)
				if err != nil {
					return err
				}
				feeds[node] = reference.Value{Shape: node.Shape, Data: values}
				nodes = append(nodes, node)
			}
			geometry := nodes[tensor.PairedExtent].Shape.Dims
			width, heads, tokens, batches := geometry[0], geometry[1], geometry[tensor.PairedExtent], geometry[tensor.TripleExtent]
			initial := b.Input("initial", dtype.F32, tensor.MustShape(width, width, heads, batches))
			feeds[initial] = reference.Value{Shape: initial.Shape, Data: make([]float32, width*width*heads*batches)}
			packed := b.GatedDeltaNetChunked(nodes[0], nodes[1], nodes[tensor.PairedExtent], nodes[tensor.TripleExtent], nodes[tensor.MaxDimensions], initial, true)
			attention := b.BF16Round(b.FlatSlice(packed, 0, width, heads, tokens, batches))
			state := b.FlatSlice(packed, width*heads*tokens*batches, width, width, heads, batches)
			if err := b.Err(); err != nil {
				return err
			}
			program, err := compileResidentStatic(ctx, runtime, "MOSS chunk "+name, nil, nil, attention, state)
			if err != nil {
				return err
			}
			defer runtime.Release(context.WithoutCancel(ctx), program)
			expectedAttention, err := safetensors.ReadF32(sources[0].Tensors[name+".delta.output"])
			if err != nil {
				return err
			}
			expectedState, err := safetensors.ReadF32(sources[1].Tensors[name+".state_f32"])
			if err != nil {
				return err
			}
			var hashes [tensor.PairedExtent][sha256.Size]byte
			for pass := range tensor.PairedExtent {
				actual, err := runtime.Execute(ctx, program, feeds)
				if err != nil {
					return err
				}
				for index, node := range []*tensor.Tensor{attention, state} {
					values, want := actual[node].Data, expectedAttention
					boundary := "attention"
					if node == state {
						boundary, want = "state", expectedState
						transposed := make([]float32, len(values))
						for group := range heads * batches {
							for k := range width {
								for v := range width {
									transposed[(group*width+k)*width+v] = values[(group*width+v)*width+k]
								}
							}
						}
						values = transposed
					}
					failed, unequal, worst := compareMOSSValues(values, want, mossStageTolerance{})
					hash := sha256.Sum256(driver.Bytes(values))
					repeated := pass == 0 || hash == hashes[index]
					hashes[index] = hash
					if err = json.NewEncoder(output).Encode(map[string]any{"stage": "conditioning-delta", "case": name, "boundary": boundary, "reference": identities[index], "pass": pass, "values": len(want), "failed": failed, "unequal": unequal, "max_abs": worst, "repeat_checked": pass != 0, "repeat_equal": repeated}); err != nil {
						return err
					}
					if failed != 0 || unequal != 0 || !repeated {
						return fmt.Errorf("MOSS chunk native %s/%s differs (%d unequal values)", name, boundary, unequal)
					}
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
