//go:build windows

package latentimage

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/graphruntime"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func verifyMOSSConditioningAttention(ctx context.Context, runtime *graphruntime.ResidentSession, root string, output io.Writer) error {
	path := filepath.Join(root, "conditioning-cuda", "conditioning-cuda-layer3.safetensors")
	id, err := artifact.IdentifyFile(artifact.KindFile, path)
	if err != nil {
		return err
	}
	if id.DigestHex() != "4d53e5d300dc7f053633de0b1aea983b65b3f6601c88f5fb6aae50f44c9fac3a" {
		return fmt.Errorf("MOSS attention reference identity differs: %s", path)
	}
	source, err := safetensors.OpenFile(path)
	if err != nil {
		return err
	}
	defer source.Close()
	for _, name := range []string{"english.0", "english.1", "chinese.0", "chinese.1", "layout.0", "layout.1"} {
		if err := func() error {
			b := tensor.NewBuilder()
			feeds := map[*tensor.Tensor]reference.Value{}
			var nodes []*tensor.Tensor
			for _, label := range []string{"query", "key", "value"} {
				shape, values, err := mossAttentionValues(source.Tensors[name+".sdpa."+label])
				if err != nil {
					return err
				}
				x := b.Input(label, dtype.F32, shape)
				feeds[x] = reference.Value{Shape: shape, Data: values}
				nodes = append(nodes, x)
			}
			// Native math SDPA scales by the inverse square root of head width.
			scale := float32(1 / math.Sqrt(float64(nodes[0].Shape.Dims[0])))
			out := b.BF16Round(b.AttentionWithOptions(nodes[0], nodes[1], nodes[tensor.PairedExtent], tensor.AttentionOptions{Scale: scale, Causal: true, NativeMath: true}))
			if err := b.Err(); err != nil {
				return err
			}
			program, err := compileResidentStatic(ctx, runtime, "MOSS attention "+name, nil, nil, out)
			if err != nil {
				return err
			}
			defer runtime.Release(context.WithoutCancel(ctx), program)
			_, want, err := mossAttentionValues(source.Tensors[name+".sdpa.output"])
			if err != nil {
				return err
			}
			var previous [sha256.Size]byte
			for pass := range tensor.PairedExtent {
				actual, err := runtime.Execute(ctx, program, feeds)
				if err != nil {
					return err
				}
				values := actual[out].Data
				failed, unequal, worst := compareMOSSValues(values, want, mossStageTolerance{})
				hash := sha256.Sum256(driver.Bytes(values))
				repeated := pass == 0 || hash == previous
				previous = hash
				if err = json.NewEncoder(output).Encode(map[string]any{"stage": "conditioning-attention", "case": name, "reference": id, "pass": pass, "values": len(want), "failed": failed, "unequal": unequal, "max_abs": worst, "repeat_checked": pass != 0, "repeat_equal": repeated}); err != nil {
					return err
				}
				if failed != 0 || unequal != 0 || !repeated {
					return fmt.Errorf("MOSS native attention %s differs (%d unequal values)", name, unequal)
				}
			}
			return nil
		}(); err != nil {
			return err
		}
	}
	return nil
}

func mossAttentionValues(item safetensors.Tensor) (tensor.Shape, []float32, error) {
	if len(item.Shape) != tensor.MaxDimensions {
		return tensor.Shape{}, nil, fmt.Errorf("MOSS attention reference %s must have rank four", item.Name)
	}
	batches, heads, tokens, width := item.Shape[0], item.Shape[1], item.Shape[tensor.PairedExtent], item.Shape[tensor.TripleExtent]
	shape, err := tensor.NewShape(width, heads, tokens, batches)
	if err != nil {
		return tensor.Shape{}, nil, err
	}
	values, err := safetensors.ReadF32(item)
	if err != nil {
		return tensor.Shape{}, nil, err
	}
	data := make([]float32, len(values))
	for b := range batches {
		for h := range heads {
			for t := range tokens {
				for d := range width {
					data[((b*tokens+t)*heads+h)*width+d] = values[((b*heads+h)*tokens+t)*width+d]
				}
			}
		}
	}
	return shape, data, nil
}
