//go:build windows

package latentimage

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"strconv"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/graphruntime"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/torchrng"
)

// mossNativeNegativeSkip is the zero-based default in the pinned pipeline and full-image producer.
// Reconsider it only with a separately captured sampling contract.
const mossNativeNegativeSkip = 22

// Native reference metadata uses decimal signed 64-bit seeds and single-precision guidance.
// These parsing widths change only with the reference metadata contract.
const (
	mossMetadataRadix = 10
	mossSeedBits      = 64
	mossGuidanceBits  = 32
)

func verifyMOSSTrajectory(ctx context.Context, runtime *graphruntime.ResidentSession, modelRoot string, spec TransformerSpec, name string, identity artifact.ID, source *safetensors.Source, tolerances map[string]mossStageTolerance, output io.Writer) error {
	var scheduler struct {
		Total    int     `json:"num_train_timesteps"`
		Shift    float64 `json:"shift"`
		Training float64 `json:"training_shift"`
	}
	if e := readJSON(filepath.Join(modelRoot, "scheduler", "scheduler_config.json"), &scheduler); e != nil {
		return e
	}
	metadata := map[string]string{}
	for _, m := range source.Metadata {
		metadata = m
	}
	steps, e := strconv.Atoi(metadata["steps"])
	if e != nil {
		return e
	}
	seed, e := strconv.ParseInt(metadata["seed"], mossMetadataRadix, mossSeedBits)
	if e != nil {
		return e
	}
	guidance, e := strconv.ParseFloat(metadata["guidance"], mossGuidanceBits)
	if e != nil {
		return e
	}
	schedule, e := compileNearestGridFlowSchedule(steps, scheduler.Total, scheduler.Shift, scheduler.Training)
	if e != nil {
		return e
	}
	settings := rotarySamplingSpec{schedule: schedule, guidance: float32(guidance), skip: []int{mossNativeNegativeSkip}, stop: 1}
	read := func(key string) ([]float32, error) {
		value, ok := source.Tensors[key]
		if !ok {
			return nil, fmt.Errorf("MOSS reference boundary %s missing", key)
		}
		return safetensors.ReadF32(value)
	}
	for key, values := range map[string][]float64{"schedule.sigmas": schedule.Sigmas, "schedule.timesteps": schedule.Timesteps} {
		want, e := read(key)
		if e != nil {
			return e
		}
		if len(values) != len(want) {
			return fmt.Errorf("MOSS %s length differs", key)
		}
		for i, v := range values {
			if float32(v) != want[i] {
				return fmt.Errorf("MOSS %s differs at %d", key, i)
			}
		}
	}
	latent := source.Tensors["initial_latents"]
	if len(latent.Shape) != rotaryCoordinateAxes || latent.Shape[0] != tensor.SingletonExtent || latent.Shape[1] != uint64(spec.InChannels) {
		return fmt.Errorf("MOSS sampling latent shape differs")
	}
	initial, e := torchrng.CPUNormal(seed, int(latent.Elements()))
	if e != nil {
		return e
	}
	positive, e := read("conditioning.0.hidden")
	if e != nil {
		return e
	}
	negative, e := read("conditioning.1.hidden")
	if e != nil {
		return e
	}
	hashes := map[string][sha256.Size]byte{}
	for pass := range tensor.PairedExtent {
		result, e := runRotarySampling(ctx, runtime, modelRoot, spec, int(latent.Shape[tensor.PairedExtent]), int(latent.Shape[tensor.TripleExtent]), initial, positive, negative, settings, func(boundary string, actual []float32) error {
			want, e := read(boundary)
			if e != nil {
				return e
			}
			if len(want) != len(actual) {
				return fmt.Errorf("MOSS %s length differs", boundary)
			}
			tolerance := tolerances[source.Tensors[boundary].DType]
			if boundary == "initial_latents" {
				tolerance = mossStageTolerance{}
			}
			failed, unequal, worst := compareMOSSValues(actual, want, tolerance)
			hash := sha256.Sum256(driver.Bytes(actual))
			repeatEqual := true
			if pass == 0 {
				hashes[boundary] = hash
			} else {
				previous, ok := hashes[boundary]
				repeatEqual = ok && previous == hash
			}
			if e := json.NewEncoder(output).Encode(map[string]any{"stage": "sampling", "case": name, "boundary": boundary, "reference": identity, "pass": pass, "values": len(want), "failed": failed, "unequal": unequal, "max_abs": worst, "repeat_checked": pass != 0, "repeat_equal": repeatEqual, "resident_weight_and_static_bytes": runtime.StaticBytes()}); e != nil {
				return e
			}
			if failed != 0 || !repeatEqual {
				return fmt.Errorf("MOSS sampling violates frozen comparison in %s/%s", name, boundary)
			}
			return nil
		})
		if e != nil {
			return e
		}
		if e := result.outputs.Release(context.WithoutCancel(ctx)); e != nil {
			return e
		}
	}
	for key := range source.Tensors {
		if strings.HasPrefix(key, "prediction.") || strings.HasPrefix(key, "velocity.") || strings.HasPrefix(key, "latents.") {
			if _, ok := hashes[key]; !ok {
				return fmt.Errorf("MOSS sampling omitted reference boundary %s", key)
			}
		}
	}
	return nil
}
