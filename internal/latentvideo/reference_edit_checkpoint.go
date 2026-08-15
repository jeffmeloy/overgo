package latentvideo

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"overgo/internal/pytorchzip"
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
		if meta.Name == "" || seen[meta.Name] {
			return plan, fmt.Errorf("reference edit checkpoint: duplicate or empty tensor %q", meta.Name)
		}
		seen[meta.Name] = true
		normalized[index] = meta
		if strings.HasPrefix(meta.Name, "blocks.") {
			remainder := strings.TrimPrefix(meta.Name, "blocks.")
			ordinal, parseErr := strconv.Atoi(strings.SplitN(remainder, ".", 2)[0])
			if parseErr == nil {
				layers[ordinal] = true
			}
		}
	}
	patch, ok := findTensor(normalized, "patch_embedding.weight")
	if !ok || len(patch.Shape) != 5 || patch.Shape[0] != int64(base.Dim) || patch.Shape[2] != int64(base.PatchSize[0]) || patch.Shape[3] != int64(base.PatchSize[1]) || patch.Shape[4] != int64(base.PatchSize[2]) {
		sample := make([]string, min(8, len(normalized)))
		for index := range sample {
			sample[index] = normalized[index].Name
		}
		return plan, fmt.Errorf("reference edit checkpoint: incompatible patch embedding %v; names=%v", patch.Shape, sample)
	}
	inputChannels := int(patch.Shape[1])
	sourceChannels := inputChannels - base.InDim
	if sourceChannels != base.InDim || len(layers) != base.NumLayers {
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
		if binding.Meta.Numel != want || binding.Meta.DType != "BFloat16Storage" && binding.Meta.DType != "FloatStorage" {
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
	reader, err := pytorchzip.Open(p.Path)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	weights := &DenoiserWeights{values: make(map[string][]float32, len(p.bindings))}
	for name, binding := range p.bindings {
		values, err := reader.ReadBinding(binding)
		if err != nil {
			return nil, fmt.Errorf("reference edit checkpoint %s: %w", name, err)
		}
		weights.values[name] = values
		weights.Bytes += int64(len(values) * 4)
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
		bytes += int64(len(values[index]) * 4)
	}
	weights.Linear0W, weights.Linear0B = values[0], values[1]
	weights.Linear2W, weights.Linear2B = values[2], values[3]
	return weights, bytes, nil
}

// ReferenceEditTextConditioning: shared encoder, checkpoint-owned projection.
func ReferenceEditTextConditioning(spec TextConditioningSpec, prompt string, checkpoint ReferenceEditCheckpoint) (TextConditioningResult, error) {
	weights, bytes, err := checkpoint.loadTextProjection()
	if err != nil {
		return TextConditioningResult{}, err
	}
	return textConditioningWithWeights(spec, prompt, weights, bytes)
}
