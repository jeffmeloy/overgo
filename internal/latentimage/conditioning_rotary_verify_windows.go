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

// These rotary parameters belong to the pinned MOSS text-encoder checkpoint.
// Reconsider them when changing the checkpoint or native fixture identity.
const (
	mossConditioningRotaryDimensions = 64
	mossConditioningRotaryBase       = 10000000
)

func verifyMOSSConditioningRotary(ctx context.Context, runtime *graphruntime.ResidentSession, root string, output io.Writer) error {
	path := filepath.Join(root, "conditioning-cuda", "conditioning-cuda-layer3.safetensors")
	id, err := artifact.IdentifyFile(artifact.KindFile, path)
	if err != nil {
		return err
	}
	if id.DigestHex() != "4d53e5d300dc7f053633de0b1aea983b65b3f6601c88f5fb6aae50f44c9fac3a" {
		return fmt.Errorf("MOSS rotary reference identity differs: %s", path)
	}
	source, err := safetensors.OpenFile(path)
	if err != nil {
		return err
	}
	defer source.Close()
	for _, name := range []string{"english.0", "english.1", "chinese.0", "chinese.1", "layout.0", "layout.1"} {
		for _, label := range []string{"query", "key"} {
			if err := func() error {
				shape, values, err := mossAttentionValues(source.Tensors[name+".rope."+label])
				if err != nil {
					return err
				}
				b := tensor.NewBuilder()
				x := b.Input(label, dtype.F32, shape)
				feeds := map[*tensor.Tensor]reference.Value{x: {Shape: shape, Data: values}}
				positions := make([]uint32, shape.Dims[tensor.PairedExtent])
				for i := range positions {
					positions[i] = uint32(i)
				}
				out := b.RoPEWithOptions(x, tensor.RoPEOptions{NativeBF16: true, Layout: tensor.RoPELayoutNeoX, Positions: positions, RotaryDimensions: mossConditioningRotaryDimensions, FrequencyBase: mossConditioningRotaryBase, FrequencyScale: tensor.UnitFrequencyScale})
				if err := b.Err(); err != nil {
					return err
				}
				program, err := compileResidentStatic(ctx, runtime, "MOSS rotary "+name+" "+label, nil, nil, out)
				if err != nil {
					return err
				}
				defer runtime.Release(context.WithoutCancel(ctx), program)
				_, want, err := mossAttentionValues(source.Tensors[name+".rope."+label+".output"])
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
					if err = json.NewEncoder(output).Encode(map[string]any{"stage": "conditioning-rotary", "case": name, "boundary": label, "reference": id, "pass": pass, "values": len(want), "failed": failed, "unequal": unequal, "max_abs": worst, "repeat_checked": pass != 0, "repeat_equal": repeated}); err != nil {
						return err
					}
					if failed != 0 || unequal != 0 || !repeated {
						return fmt.Errorf("MOSS native rotary %s/%s differs (%d unequal values)", name, label, unequal)
					}
				}
				return nil
			}(); err != nil {
				return err
			}
		}
	}
	return nil
}
