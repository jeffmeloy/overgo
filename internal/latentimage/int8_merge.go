package latentimage

import (
	"fmt"
	"hash/crc32"

	"overgo/internal/comfyquant"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// comfyModulePrefix opens the name ComfyUI gives a diffusion model's module, which seeds the stochastic rounding of a
// weight it merges a LoRA into (comfy/model_patcher.py sets each module's seed_key to its name in the model).
const comfyModulePrefix = "diffusion_model."

// bf16Bytes is the bytes of a BF16 value.
const bf16Bytes = 2

// adapterPair is one adapter's pair over a projection, at the strength a request applies the adapter.
type adapterPair struct {
	adapter  *mediaAdapter
	patch    lowRankPatch
	strength float32
}

// pairPlacer places an adapter's tensor of name, rows of columns values in BF16 (bf16Tensor), on the card and gives
// where it lies, once for every merge that reads it (bindMerged keeps it in the session under the adapter's content).
type pairPlacer func(adapter *mediaAdapter, name string, rows, columns uint64) (driver.DevicePtr, error)

// convRotMerge is the merge on the card (executor.Int8ConvRotMerge) of pairs into a joint projection the int8 ConvRot
// file holds in int8, as ComfyUI merges a LoRA into it while its module casts it (moss-int8-adapters): the file's int8
// rows and scales for weight; each pair's up rows over the weight's rows and its down, in BF16 as torch casts them, at
// its factor, in the order the adapters apply, read where place puts them; and the stochastic rounding of the module
// the file stores the weight in, seeded by the CRC-32 of its name in ComfyUI's model (comfy.utils.string_to_seed), at
// the weight's first row in it. A weight held on the card (held, where set) is read there, and its rows are not read
// from the file.
func convRotMerge(source *safetensors.Source, weight string, pairs []adapterPair, place pairPlacer, held driver.DevicePtr) (executor.Int8ConvRotMerge, error) {
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
	if held != 0 {
		merge.Held = held
	} else if merge.Stored, err = stored.Stored(); err != nil {
		return executor.Int8ConvRotMerge{}, err
	}
	for _, pair := range pairs {
		patch := pair.patch
		if patch.rows != rows || patch.in != inner {
			return executor.Int8ConvRotMerge{}, fmt.Errorf("%s: a pair over %d rows of %d is no pair over the weight's %d of %d", weight, patch.rows, patch.in, rows, inner)
		}
		up, down, err := patch.onCard(pair.adapter, place)
		if err != nil {
			return executor.Int8ConvRotMerge{}, fmt.Errorf("%s: %w", weight, err)
		}
		merge.Pairs = append(merge.Pairs, executor.Int8ConvRotPair{Up: up, Down: down, Rank: uint32(patch.rank), Factor: patch.factor(pair.strength)})
	}
	return merge, nil
}

// onCard is where the pair's up rows and down lie on the card in BF16, each of the adapter's tensors placed whole, a
// fused qkv's up for its q, k and v alike, and the pair's rows read from it.
func (patch lowRankPatch) onCard(adapter *mediaAdapter, place pairPlacer) (up, down driver.DevicePtr, err error) {
	rows := patch.rows
	if patch.split {
		rows *= tensor.TripleExtent
	}
	if up, err = place(adapter, patch.up, rows, patch.rank); err != nil {
		return 0, 0, err
	}
	if down, err = place(adapter, patch.down, patch.rank, patch.in); err != nil {
		return 0, 0, err
	}
	return up + driver.DevicePtr(bf16Bytes*patch.from*patch.rank), down, nil
}

// bf16Tensor is the adapter file's tensor of name in BF16, as a merge's pair reads it, the file opened once while the
// sampling merges (release closes it).
func (a *mediaAdapter) bf16Tensor(name string) ([]byte, error) {
	if a.source == nil {
		source, err := safetensors.OpenFile(a.path)
		if err != nil {
			return nil, err
		}
		a.source = source
	}
	value, found := a.source.Tensors[name]
	if !found {
		return nil, fmt.Errorf("adapter %s holds no %s", a.path, name)
	}
	return weightPayload(value, dtype.BF16)
}

// release closes the adapter's file once the sampling that merged from it ends: the tensors its merges read are kept on
// the card (bindMerged), and the 819 MiB text to image Turbo's two adapters rounded had stayed reachable on the host
// after its picture (SQA-439).
func (a *mediaAdapter) release() (err error) {
	if a.source != nil {
		err = a.source.Close()
	}
	a.source = nil
	return err
}
