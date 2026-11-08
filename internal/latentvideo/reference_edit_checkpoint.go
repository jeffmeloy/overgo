package latentvideo

import (
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/pytorchzip"
	"overgo/internal/tensor"
)

type referenceEditCheckpoint struct {
	Path           string
	Config         DenoiserConfig
	SourceChannels int
	Layers         int
	bindings       map[string]pytorchzip.TensorBinding
}

// CompileReferenceEditCheckpoint: normalize and bind LiveEdit onto Wan.
func CompileReferenceEditCheckpoint(path string, base DenoiserConfig) (referenceEditCheckpoint, error) {
	var plan referenceEditCheckpoint
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
		if after, ok := strings.CutPrefix(meta.Name, "blocks."); ok {
			remainder := after
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
	// A LiveEdit editor reads a source clip's latent beside the noise, as many channels again; a text-to-video student
	// (Self-Forcing) reads the noise alone.
	sourceChannels := inputChannels - base.InDim
	if !checked.Equal(sourceChannels, base.InDim) && checked.Nonzero(sourceChannels) {
		return plan, fmt.Errorf("reference edit checkpoint: input=%d source=%d, want source=%d or none", inputChannels, sourceChannels, base.InDim)
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
	slices.Sort(names)
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
	return referenceEditCheckpoint{Path: path, Config: live, SourceChannels: sourceChannels, Layers: len(layers), bindings: byName}, nil
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

// student says the checkpoint generates from text alone, reading no source clip.
func (p referenceEditCheckpoint) student() bool {
	return !checked.Nonzero(p.SourceChannels)
}

// Binding: immutable named checkpoint binding.
func (p referenceEditCheckpoint) Binding(name string) (pytorchzip.TensorBinding, bool) {
	binding, ok := p.bindings[name]
	return binding, ok
}

// LoadWeights: stream the compiled checkpoint into shared denoiser storage.
func (p referenceEditCheckpoint) LoadWeights() (*DenoiserWeights, error) {
	return p.LoadDenoiserWeights(p.Layers)
}

// LoadDenoiserWeights reads the common tensors plus the requested block prefix, every one decoded.
func (p referenceEditCheckpoint) LoadDenoiserWeights(layers int) (*DenoiserWeights, error) {
	weights, err := p.OpenDenoiserWeights(layers)
	if err != nil {
		return nil, err
	}
	err = weights.each(func(name string, values []float32) error {
		weights.values[name] = values
		return nil
	})
	return weights, errors.Join(err, weights.Close())
}

// OpenDenoiserWeights binds the common tensors plus the requested block prefix
// and decodes the timestep conditioning's; each other tensor is decoded as it
// is taken, from the checkpoint the set keeps open until Close.
func (p referenceEditCheckpoint) OpenDenoiserWeights(layers int) (_ *DenoiserWeights, err error) {
	if !checked.PositiveInts(layers) || !checked.AtMostInt(layers, p.Layers) {
		return nil, fmt.Errorf("reference edit checkpoint: layers=%d outside 1..%d", layers, p.Layers)
	}
	reader, err := pytorchzip.Open(p.Path)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, reader.Close())
		}
	}()
	config := p.Config
	config.NumLayers = layers
	lengths := denoiserTensorLengths(config)
	weights := &DenoiserWeights{values: make(map[string][]float32, len(timestepTensors)), close: reader.Close}
	read := func(name string) ([]float32, error) {
		values, err := reader.ReadBinding(p.bindings[name])
		if err != nil {
			return nil, fmt.Errorf("reference edit checkpoint %s: %w", name, err)
		}
		return values, nil
	}
	for name, elements := range lengths {
		if _, ok := p.bindings[name]; !ok {
			return nil, fmt.Errorf("reference edit checkpoint: denoiser tensor %s is absent", name)
		}
		weights.Bytes += int64(elements * binaryschema.Uint32Bytes)
		if !slices.Contains(timestepTensors, name) {
			weights.names = append(weights.names, name)
		} else if weights.values[name], err = read(name); err != nil {
			return nil, err
		}
	}
	slices.Sort(weights.names)
	weights.open = read
	return weights, nil
}

func (p referenceEditCheckpoint) loadTextProjection() (projectionWeights, int64, error) {
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
