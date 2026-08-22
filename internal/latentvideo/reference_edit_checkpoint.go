package latentvideo

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/pytorchzip"
	"overgo/internal/tensor"
)

type ReferenceEditCheckpoint struct {
	Path           string
	Config         DenoiserConfig
	SourceChannels int
	Layers         int
	bindings       map[string]pytorchzip.TensorBinding
}

// CompileReferenceEditCheckpoint: normalize and bind LiveEdit onto Wan.
func CompileReferenceEditCheckpoint(path string, base DenoiserConfig) (ReferenceEditCheckpoint, error) {
	var plan ReferenceEditCheckpoint
	if err := base.validate(); err != nil {
		return plan, err
	}
	catalog, err := pytorchzip.ReadCatalog(path)
	if err != nil {
		return plan, fmt.Errorf("reference edit checkpoint: %w", err)
	}
	metas := catalog.Tensors
	normalized := make([]pytorchzip.TensorMeta, len(metas))
	seen := make(map[string]bool, len(metas))
	layers := make(map[int]bool)
	for index, meta := range metas {
		meta.Name = normalizeReferenceEditName(meta.Name)
		if !checked.Nonzero(meta.Name) || seen[meta.Name] {
			return plan, fmt.Errorf("reference edit checkpoint: duplicate or empty tensor %q", meta.Name)
		}
		seen[meta.Name] = true
		normalized[index] = meta
		if strings.HasPrefix(meta.Name, "blocks.") {
			remainder := strings.TrimPrefix(meta.Name, "blocks.")
			ordinal, parseErr := strconv.Atoi(strings.SplitN(remainder, ".", tensor.PairedExtent)[tensor.FirstOffset])
			if parseErr == nil {
				layers[ordinal] = true
			}
		}
	}
	patch, ok := findTensor(normalized, "patch_embedding.weight")
	if !ok {
		return plan, fmt.Errorf("reference edit checkpoint: patch embedding is absent")
	}
	patchShape, err := pytorchzip.HostShape(patch, tensor.MaxDimensions+tensor.SingletonExtent)
	if err != nil {
		return plan, fmt.Errorf("reference edit checkpoint: patch embedding: %w", err)
	}
	patchExtent := [tensor.TripleExtent]int{
		patchShape[tensor.PairedExtent],
		patchShape[tensor.TripleExtent],
		patchShape[tensor.MaxDimensions],
	}
	if !checked.Equal(patchShape[tensor.FirstOffset], base.Dim) {
		return plan, fmt.Errorf("reference edit checkpoint: patch output %d differs from dim %d", patchShape[tensor.FirstOffset], base.Dim)
	}
	if !checked.Equal(patchExtent, base.PatchSize) {
		sample := make([]string, min(tensor.PairedExtent*tensor.MaxDimensions, len(normalized)))
		for index := range sample {
			sample[index] = normalized[index].Name
		}
		return plan, fmt.Errorf("reference edit checkpoint: incompatible patch embedding %v; names=%v", patch.Shape, sample)
	}
	inputChannels := patchShape[tensor.SingletonExtent]
	sourceChannels := inputChannels - base.InDim
	if !checked.Equal(sourceChannels, base.InDim) {
		return plan, fmt.Errorf("reference edit checkpoint: input=%d source=%d, want source=%d", inputChannels, sourceChannels, base.InDim)
	}
	if !checked.Equal(len(layers), base.NumLayers) {
		return plan, fmt.Errorf("reference edit checkpoint: input=%d source=%d layers=%d, want source=%d layers=%d", inputChannels, sourceChannels, len(layers), base.InDim, base.NumLayers)
	}
	live := base
	live.InDim = inputChannels
	lengths := denoiserTensorLengths(live)
	names := make([]string, 0, len(lengths))
	for name := range lengths {
		names = append(names, name)
	}
	sort.Strings(names)
	bindings, err := pytorchzip.CompileBindings(normalized, names)
	if err != nil {
		return plan, fmt.Errorf("reference edit checkpoint: %w", err)
	}
	byName := make(map[string]pytorchzip.TensorBinding, len(bindings))
	for index, binding := range bindings {
		want := int64(lengths[names[index]])
		if !checked.Equal(binding.Meta.Numel, want) ||
			!checked.Equal(binding.Meta.DType, pytorchzip.BFloat16StorageClass()) && !checked.Equal(binding.Meta.DType, pytorchzip.Float32StorageClass()) {
			return plan, fmt.Errorf("reference edit checkpoint: %s dtype=%s elements=%d want=%d", names[index], binding.Meta.DType, binding.Meta.Numel, want)
		}
		byName[names[index]] = binding
	}
	textBindings, err := pytorchzip.CompileBindings(normalized, projectionTensorNames[:])
	if err != nil {
		return plan, fmt.Errorf("reference edit checkpoint text projection: %w", err)
	}
	for index, binding := range textBindings {
		byName[projectionTensorNames[index]] = binding
	}
	return ReferenceEditCheckpoint{Path: path, Config: live, SourceChannels: sourceChannels, Layers: len(layers), bindings: byName}, nil
}

func normalizeReferenceEditName(name string) string {
	for _, prefix := range []string{"_fsdp_wrapped_module.", "module.", "model."} {
		name = strings.TrimPrefix(name, prefix)
	}
	return name
}

func findTensor(metas []pytorchzip.TensorMeta, name string) (pytorchzip.TensorMeta, bool) {
	for _, meta := range metas {
		if meta.Name == name {
			return meta, true
		}
	}
	return pytorchzip.TensorMeta{}, false
}

// Binding: immutable named checkpoint binding.
func (p ReferenceEditCheckpoint) Binding(name string) (pytorchzip.TensorBinding, bool) {
	binding, ok := p.bindings[name]
	return binding, ok
}

// LoadWeights: stream the compiled checkpoint into shared denoiser storage.
func (p ReferenceEditCheckpoint) LoadWeights() (*DenoiserWeights, error) {
	return p.LoadDenoiserWeights(p.Layers)
}

// LoadDenoiserWeights streams common tensors plus the requested block prefix.
func (p ReferenceEditCheckpoint) LoadDenoiserWeights(layers int) (*DenoiserWeights, error) {
	if !checked.PositiveInts(layers) || !checked.AtMostInt(layers, p.Layers) {
		return nil, fmt.Errorf("reference edit checkpoint: layers=%d outside 1..%d", layers, p.Layers)
	}
	reader, err := pytorchzip.Open(p.Path)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	config := p.Config
	config.NumLayers = layers
	lengths := denoiserTensorLengths(config)
	weights := &DenoiserWeights{values: make(map[string][]float32, len(lengths))}
	for name := range lengths {
		binding, ok := p.bindings[name]
		if !ok {
			return nil, fmt.Errorf("reference edit checkpoint: denoiser tensor %s is absent", name)
		}
		values, err := reader.ReadBinding(binding)
		if err != nil {
			return nil, fmt.Errorf("reference edit checkpoint %s: %w", name, err)
		}
		weights.values[name] = values
		weights.Bytes += int64(len(values) * binaryschema.Uint32Bytes)
	}
	return weights, nil
}

func (p ReferenceEditCheckpoint) loadTextProjection() (projectionWeights, int64, error) {
	var weights projectionWeights
	reader, err := pytorchzip.Open(p.Path)
	if err != nil {
		return weights, 0, err
	}
	defer reader.Close()
	values := make([][]float32, len(projectionTensorNames))
	var bytes int64
	for index, name := range projectionTensorNames {
		binding, ok := p.bindings[name]
		if !ok {
			return weights, 0, fmt.Errorf("reference edit checkpoint: text projection %s is absent", name)
		}
		values[index], err = reader.ReadBinding(binding)
		if err != nil {
			return weights, 0, err
		}
		bytes += int64(len(values[index]) * binaryschema.Uint32Bytes)
	}
	weights.Linear0W, weights.Linear0B = values[tensor.FirstOffset], values[tensor.SingletonExtent]
	weights.Linear2W, weights.Linear2B = values[tensor.PairedExtent], values[tensor.TripleExtent]
	return weights, bytes, nil
}
