package latentimage

import (
	"errors"
	"fmt"
	"hash/crc32"

	"overgo/internal/comfyquant"
	"overgo/internal/cuda/executor"
	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
)

// comfyModulePrefix opens the name ComfyUI gives a diffusion model's module, which seeds the stochastic rounding of a
// weight it merges a LoRA into (comfy/model_patcher.py sets each module's seed_key to its name in the model).
const comfyModulePrefix = "diffusion_model."

// adapterPair is one adapter's pair over a projection, at the strength a request applies the adapter.
type adapterPair struct {
	adapter  *mediaAdapter
	patch    lowRankPatch
	strength float32
}

// convRotMerge is the merge on the card (executor.Int8ConvRotMerge) of pairs into a joint projection the int8 ConvRot
// file holds in int8, as ComfyUI merges a LoRA into it while its module casts it (moss-int8-adapters): the file's int8
// rows and scales for weight; each pair's up rows over the weight's rows and its down, in BF16 as torch casts them, at
// its factor, in the order the adapters apply; and the stochastic rounding of the module the file stores the weight in,
// seeded by the CRC-32 of its name in ComfyUI's model (comfy.utils.string_to_seed), at the weight's first row in it.
func convRotMerge(source *safetensors.Source, weight string, pairs []adapterPair) (executor.Int8ConvRotMerge, error) {
	stored, quantized, err := comfyquant.Find(source, weight)
	if err != nil {
		return executor.Int8ConvRotMerge{}, err
	}
	if !quantized || stored.Group() != comfyquant.Group {
		return executor.Int8ConvRotMerge{}, fmt.Errorf("%s is no int8 ConvRot weight over groups of %d columns", weight, comfyquant.Group)
	}
	shape := stored.Shape()
	rows, inner := shape[0], shape[1]
	module, first := stored.Module()
	merge := executor.Int8ConvRotMerge{Rows: uint32(rows), Inner: uint32(inner), FirstRow: first,
		Seed: uint64(crc32.ChecksumIEEE([]byte(comfyModulePrefix + module)))}
	if merge.Stored, err = stored.Stored(); err != nil {
		return executor.Int8ConvRotMerge{}, err
	}
	for _, pair := range pairs {
		patch := pair.patch
		if patch.rows != rows || patch.in != inner {
			return executor.Int8ConvRotMerge{}, fmt.Errorf("%s: a pair over %d rows of %d is no pair over the weight's %d of %d", weight, patch.rows, patch.in, rows, inner)
		}
		up, down, err := pairPayload(pair.adapter.path, patch)
		if err != nil {
			return executor.Int8ConvRotMerge{}, fmt.Errorf("%s: %w", weight, err)
		}
		merge.Pairs = append(merge.Pairs, executor.Int8ConvRotPair{Up: up, Down: down, Rank: uint32(patch.rank), Factor: patch.factor(pair.strength)})
	}
	return merge, nil
}

// pairPayload reads a pair's up rows and down from its adapter's file in BF16.
func pairPayload(path string, patch lowRankPatch) (up, down []byte, err error) {
	source, err := safetensors.OpenFile(path)
	if err != nil {
		return nil, nil, err
	}
	defer func() { err = errors.Join(err, source.Close()) }()
	upValue, upHeld := source.Tensors[patch.up]
	downValue, downHeld := source.Tensors[patch.down]
	if !upHeld || !downHeld {
		return nil, nil, fmt.Errorf("adapter %s holds no %s or %s", path, patch.up, patch.down)
	}
	if up, err = weightPayload(upValue, dtype.BF16); err != nil {
		return nil, nil, err
	}
	if down, err = weightPayload(downValue, dtype.BF16); err != nil {
		return nil, nil, err
	}
	width := 2 * patch.rank
	return up[patch.from*width : (patch.from+patch.rows)*width], down, nil
}
