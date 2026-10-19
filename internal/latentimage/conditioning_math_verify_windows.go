//go:build windows

package latentimage

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/graphruntime"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func verifyMOSSConditioningMath(ctx context.Context, runtime *graphruntime.ResidentSession, root string, output io.Writer) error {
	path := filepath.Join(root, "conditioning-cuda", "conditioning-cuda-precision.safetensors")
	id, err := artifact.IdentifyFile(artifact.KindFile, path)
	if err != nil {
		return err
	}
	if id.DigestHex() != "c57c9f33e59b5f21d745ef8f81bb4574148be2549da5f226e2a1934c234ce655" {
		return fmt.Errorf("MOSS precision fixture identity differs: %s", path)
	}
	source, err := safetensors.OpenFile(path)
	if err != nil {
		return err
	}
	defer source.Close()
	for _, name := range []string{"english.0", "english.1", "chinese.0", "chinese.1", "layout.0", "layout.1"} {
		err := func() error {
			b := tensor.NewBuilder()
			feeds := map[*tensor.Tensor]reference.Value{}
			input := func(label string) (*tensor.Tensor, error) {
				item, ok := source.Tensors[name+"."+label]
				if !ok {
					return nil, fmt.Errorf("MOSS precision input %q missing", label)
				}
				dims := slices.Clone(item.Shape)
				slices.Reverse(dims)
				shape, err := tensor.NewShape(dims...)
				if err != nil {
					return nil, err
				}
				values, err := safetensors.ReadF32(item)
				if err != nil {
					return nil, err
				}
				node := b.Input(label, dtype.F32, shape)
				feeds[node] = reference.Value{Shape: shape, Data: values}
				return node, nil
			}
			var nodes []*tensor.Tensor
			var labels []string
			for _, label := range []string{"input_layernorm", "post_attention_layernorm", "linear_attn.norm"} {
				x, err := input(label + ".input")
				if err != nil {
					return err
				}
				nodes = append(nodes, b.RMSNormNative(x, mossRefinementEpsilon))
				labels = append(labels, label+".normalized")
			}
			alpha, err := input("gate.alpha")
			if err != nil {
				return err
			}
			a, err := input("gate.A_log")
			if err != nil {
				return err
			}
			bias, err := input("gate.dt_bias")
			if err != nil {
				return err
			}
			gate := b.Multiply(b.Scale(b.ExpNative(a), -1), b.SoftplusNative(b.Add(alpha, bias)))
			nodes = append(nodes, gate)
			labels = append(labels, "gate.output")
			if b.Err() != nil {
				return b.Err()
			}
			program, err := compileResidentStatic(ctx, runtime, "MOSS native math "+name, nil, nil, nodes...)
			if err != nil {
				return err
			}
			defer runtime.Release(context.WithoutCancel(ctx), program)
			hashes := make([][sha256.Size]byte, len(nodes))
			for pass := range tensor.PairedExtent {
				actual, err := runtime.Execute(ctx, program, feeds)
				if err != nil {
					return err
				}
				for i, node := range nodes {
					want, err := safetensors.ReadF32(source.Tensors[name+"."+labels[i]])
					if err != nil {
						return err
					}
					values := actual[node].Data
					failed, unequal, worst := compareMOSSValues(values, want, mossStageTolerance{})
					hash := sha256.Sum256(driver.Bytes(values))
					repeated := pass == 0 || hash == hashes[i]
					hashes[i] = hash
					if err = json.NewEncoder(output).Encode(map[string]any{"stage": "conditioning-math", "case": name, "boundary": labels[i], "reference": id, "pass": pass, "values": len(want), "failed": failed, "unequal": unequal, "max_abs": worst, "repeat_checked": pass != 0, "repeat_equal": repeated}); err != nil {
						return err
					}
					if failed != 0 || unequal != 0 || !repeated {
						return fmt.Errorf("MOSS native math %s/%s differs (%d unequal values)", name, labels[i], unequal)
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
