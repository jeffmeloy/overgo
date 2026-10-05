package densecausal

import (
	"errors"
	"fmt"
	"math"
	"strings"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/quant"
	"overgo/internal/tensorcatalog"
)

// llamaArchitecture is the GGUF architecture whose conversion permutes the
// query and key rows for the interleaved rotary layout.
const llamaArchitecture = "llama"

// The HF names the permuted projections end with.
const (
	queryProjection = ".self_attn.q_proj.weight"
	keyProjection   = ".self_attn.k_proj.weight"
	// rotaryPair is how many rows a rotary pair spans: the conversion interleaves
	// each head's first half of rows with its second.
	rotaryPair = 2
)

// unpermuteRows undoes the permutation GGUF conversion applies to a query or
// key projection's output rows: each of heads heads stores its rows interleaved
// in rotary pairs (row 2j+k) where the checkpoint holds its first half of rows,
// then its second (row k*half+j).
func unpermuteRows(values []float32, rows, columns, heads int) []float32 {
	half := rows / heads / rotaryPair
	restored := make([]float32, len(values))
	for head := range heads {
		first := head * rotaryPair * half
		for j := range half {
			for k := range rotaryPair {
				stored, original := first+j*rotaryPair+k, first+k*half+j
				copy(restored[original*columns:(original+1)*columns], values[stored*columns:(stored+1)*columns])
			}
		}
	}
	return restored
}

// LoadGGUF opens one GGUF model file and materializes the same trainable
// F32 inventory Load materializes from a safetensors directory: GGUF is a
// different representation of the same weights, so every tensor
// dequantizes to f32 through the quantization owner, names map back to
// the HF catalog through the same table conversion writes with, and the
// model facts read from GGUF metadata. The llama architecture's conversion
// permutes the query and key rows for the interleaved rotary layout, and
// the load undoes it; another interleaved-rotary architecture refuses until
// its conversion's permutation is established, and mixture models refuse
// until the expert name mapping lands — an exact refusal, never a silently
// wrong catalog.
func LoadGGUF(path string) (*Model, error) {
	return loadGGUF(path, nil)
}

// LoadGGUFFrozen is LoadGGUF for a model whose weights hold names, by their
// catalog names, stay in the quantization the file stores them in instead of
// dequantizing at load: they are dequantized into a scratch buffer as each layer
// reads them, so the model holds their stored bytes and one layer's scratch, not
// every weight in full precision. A tensor the file stores in full precision is
// loaded as LoadGGUF loads it, and so is a llama query or key projection. The model is for training an adapter over those
// weights (see Model.OnDequantize); it trains no weight of its own.
func LoadGGUFFrozen(path string, hold func(name string) bool) (*Model, error) {
	if hold == nil {
		return nil, errors.New("densecausal: freezing GGUF weights needs the predicate that names them")
	}
	return loadGGUF(path, hold)
}

func loadGGUF(path string, hold func(name string) bool) (*Model, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	spec, err := model.ReadSpec(file)
	if err != nil {
		return nil, fmt.Errorf("densecausal: read GGUF spec: %w", err)
	}
	// A profile that declares the normal (interleaved) rotary layout is one
	// whose GGUF conversion permuted the query and key rows from the HF
	// layout the trainer indexes; the llama conversion's permutation is
	// known, so its rows are put back as they load.
	permuted := spec.Profile().Position == model.PositionNormal
	if permuted && spec.Architecture != llamaArchitecture {
		return nil, fmt.Errorf("densecausal: %s declares the normal rotary layout, so its GGUF stores rope-permuted attention rows; training load requires its conversion's unpermutation, which is established for %s alone", spec.Architecture, llamaArchitecture)
	}
	if spec.HasExperts() {
		return nil, errors.New("densecausal: mixture GGUF training load requires the expert tensor mapping, which is not yet implemented")
	}
	weights := make(map[string][]float32, len(file.Tensors))
	shapes := make(map[string][]int, len(file.Tensors))
	held := map[string]frozenTensor{}
	for _, info := range file.Tensors {
		if info.Name == "rope_freqs.weight" {
			continue
		}
		name, ok := tensorcatalog.HFDenseTensorName(info.Name)
		if !ok {
			return nil, fmt.Errorf("densecausal: GGUF tensor %q has no dense training mapping", info.Name)
		}
		extents := info.Extents()
		shape := make([]int, len(extents))
		elements := uint64(1)
		for index, extent := range extents {
			// GGUF extents are fastest-first; the HF catalog shape is the
			// same memory layout with the axes reversed.
			shape[len(extents)-1-index] = int(extent)
			elements *= extent
		}
		if info.Size > uint64(math.MaxInt) {
			return nil, fmt.Errorf("densecausal: tensor %q exceeds addressable memory", info.Name)
		}
		storage := make([]byte, int(info.Size))
		if err := file.ReadTensorData(info, storage); err != nil {
			return nil, fmt.Errorf("densecausal: read %q: %w", info.Name, err)
		}
		shapes[name] = shape
		// A held tensor is dequantized as stored, so a llama query or key
		// projection, whose stored rows the load must reorder, is read in full.
		reordered := permuted && (strings.HasSuffix(name, queryProjection) || strings.HasSuffix(name, keyProjection))
		if hold != nil && info.Type != gguf.DTypeF32 && hold(name) && !reordered {
			held[name] = frozenTensor{stored: info.Type, data: storage}
			continue
		}
		values, err := quant.Dequantize(info.Type, storage, elements)
		if err != nil {
			return nil, fmt.Errorf("densecausal: dequantize %q: %w", info.Name, err)
		}
		if permuted {
			switch {
			case strings.HasSuffix(name, queryProjection):
				values = unpermuteRows(values, shape[0], shape[1], int(spec.HeadCount))
			case strings.HasSuffix(name, keyProjection):
				values = unpermuteRows(values, shape[0], shape[1], int(spec.HeadCountKV))
			}
		}
		weights[name] = values
	}
	// A frozen projection's slice, which a layer binds, is a prefix of the scratch of
	// its kind, shared by every layer, sized for the largest.
	scratch := map[string][]float32{}
	for name := range held {
		kind, size := projectionKind(name), 1
		for _, extent := range shapes[name] {
			size *= extent
		}
		if len(scratch[kind]) < size {
			scratch[kind] = make([]float32, size)
		}
	}
	for name := range held {
		size := 1
		for _, extent := range shapes[name] {
			size *= extent
		}
		weights[name] = scratch[projectionKind(name)][:size]
	}
	m, err := NewMixtureModel(
		weights, shapes, int(spec.HeadCount), int(spec.KeyLength),
		float64(spec.RopeFrequencyBase), float64(spec.RMSNormEpsilon),
		MoERouterPolicy{}, int(spec.SlidingWindow),
	)
	if err != nil {
		return nil, err
	}
	m.Dims.ContextLength = int(spec.ContextLength)
	if len(held) != 0 {
		for name := range held {
			delete(m.Weights, name)
		}
		m.frozen = &frozenWeights{tensors: held, scratch: scratch}
	}
	return m, nil
}
