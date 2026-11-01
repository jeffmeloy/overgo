//go:build windows

package latentimage

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/graphruntime"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"path/filepath"
	"slices"
)

func verifyMOSSConditioningL2(ctx context.Context, runtime *graphruntime.ResidentSession, root string, output io.Writer) error {
	path := filepath.Join(root, "conditioning-cuda", "l2-cuda.safetensors")
	id, err := artifact.IdentifyFile(artifact.KindFile, path)
	if err != nil {
		return err
	}
	if id.DigestHex() != "c57437804e63dd1de7feb69b6bbe1b0e2e0405c179a7b5e3480a8139cf59e229" {
		return fmt.Errorf("MOSS L2 fixture identity differs: %s", path)
	}
	s, err := safetensors.OpenFile(path)
	if err != nil {
		return err
	}
	defer s.Close()
	policies := map[string]float32{}
	for _, metadata := range s.Metadata {
		if err = json.Unmarshal([]byte(metadata["policies"]), &policies); err != nil {
			return err
		}
	}
	var names []string
	for name := range policies {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		err := func() error {
			b := tensor.NewBuilder()
			dims := slices.Clone(s.Tensors[name+".input"].Shape)
			slices.Reverse(dims)
			shape, err := tensor.NewShape(dims...)
			if err != nil {
				return err
			}
			x := b.Input("input", dtype.F32, shape)
			out := b.L2NormNativeBF16(x, policies[name])
			if b.Err() != nil {
				return b.Err()
			}
			values, err := safetensors.ReadF32(s.Tensors[name+".input"])
			if err != nil {
				return err
			}
			want, err := safetensors.ReadF32(s.Tensors[name+".output"])
			if err != nil {
				return err
			}
			feeds := map[*tensor.Tensor]reference.Value{x: {Shape: shape, Data: values}}
			program, err := compileResidentStatic(ctx, runtime, "MOSS native L2 "+name, nil, nil, out)
			if err != nil {
				return err
			}
			defer runtime.Release(context.WithoutCancel(ctx), program)
			var previous [sha256.Size]byte
			for pass := range tensor.PairedExtent {
				actual, err := runtime.Execute(ctx, program, feeds)
				if err != nil {
					return err
				}
				got := actual[out].Data
				failed, unequal, worst := compareMOSSValues(got, want, mossStageTolerance{})
				hash := sha256.Sum256(driver.Bytes(got))
				repeated := pass == 0 || hash == previous
				previous = hash
				if err = json.NewEncoder(output).Encode(map[string]any{"stage": "conditioning-l2", "case": name, "reference": id, "pass": pass, "values": len(want), "failed": failed, "unequal": unequal, "max_abs": worst, "repeat_checked": pass != 0, "repeat_equal": repeated}); err != nil {
					return err
				}
				if failed != 0 || unequal != 0 || !repeated {
					return fmt.Errorf("MOSS native L2 %s differs (%d unequal values)", name, unequal)
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
