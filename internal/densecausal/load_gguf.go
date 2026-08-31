package densecausal

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/gguf"
	"overgo/internal/hfgguf"
	"overgo/internal/model"
	"overgo/internal/quant"
)

// LoadGGUF opens one GGUF model file and materializes the same trainable
// F32 inventory Load materializes from a safetensors directory: GGUF is a
// different representation of the same weights, so every tensor
// dequantizes to f32 through the quantization owner, names map back to
// the HF catalog through the same table conversion writes with, and the
// model facts read from GGUF metadata. Interleaved-RoPE conversions store
// permuted attention rows and refuse until row unpermutation lands, and
// mixture models refuse until the expert name mapping lands — an exact
// refusal, never a silently wrong catalog.
func LoadGGUF(path string) (*Model, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	spec, err := model.ReadSpec(file)
	if err != nil {
		return nil, fmt.Errorf("densecausal: read GGUF spec: %w", err)
	}
	if spec.Architecture == "llama" {
		return nil, errors.New("densecausal: llama-family GGUF stores rope-permuted attention rows; training load requires unpermutation, which is not yet implemented")
	}
	if spec.HasExperts() {
		return nil, errors.New("densecausal: mixture GGUF training load requires the expert tensor mapping, which is not yet implemented")
	}
	weights := make(map[string][]float32, len(file.Tensors))
	shapes := make(map[string][]int, len(file.Tensors))
	for _, info := range file.Tensors {
		if info.Name == "rope_freqs.weight" {
			continue
		}
		name, ok := hfgguf.HFDenseTensorName(info.Name)
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
		values, err := quant.Dequantize(info.Type, storage, elements)
		if err != nil {
			return nil, fmt.Errorf("densecausal: dequantize %q: %w", info.Name, err)
		}
		weights[name] = values
		shapes[name] = shape
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
	return m, nil
}
