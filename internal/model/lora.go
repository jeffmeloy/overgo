package model

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor/reference"
)

// LoRAWeight: validated adapter pair for one base tensor.
type LoRAWeight struct {
	A, B      reference.Value
	Embedding bool
}

// LoRAAdapter: immutable loaded GGUF adapter.
type LoRAAdapter struct {
	Path             string
	Alpha            float32
	Weights          map[string]LoRAWeight
	InvocationTokens []uint32
}

// LoadLoRA: loads and validates one pinned-format GGUF adapter.
func LoadLoRA(ctx context.Context, path string, base *gguf.File, spec Spec) (*LoRAAdapter, error) {
	if path == "" {
		return nil, errors.New("LoRA path is empty")
	}
	if base == nil {
		return nil, errors.New("LoRA base model is nil")
	}
	file, err := gguf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open LoRA adapter: %w", err)
	}
	defer file.Close()
	requireString := func(key string) (string, error) {
		value, ok := file.MetadataValue(key)
		if !ok || value.Type != gguf.ValueTypeString {
			return "", fmt.Errorf("LoRA metadata %q is missing or not string", key)
		}
		result, ok := value.Data.(string)
		if !ok {
			return "", fmt.Errorf("LoRA metadata %q has invalid string storage", key)
		}
		return result, nil
	}
	generalType, err := requireString("general.type")
	if err != nil {
		return nil, err
	}
	if generalType != "adapter" {
		return nil, fmt.Errorf("LoRA general.type is %q, want adapter", generalType)
	}
	architecture, err := requireString("general.architecture")
	if err != nil {
		return nil, err
	}
	if architecture != spec.Architecture {
		return nil, fmt.Errorf("LoRA architecture %q does not match model %q", architecture, spec.Architecture)
	}
	adapterType, err := requireString("adapter.type")
	if err != nil {
		return nil, err
	}
	if adapterType != "lora" {
		return nil, fmt.Errorf("adapter.type is %q, want lora", adapterType)
	}
	result := &LoRAAdapter{Path: path, Weights: make(map[string]LoRAWeight)}
	if value, ok := file.MetadataValue("adapter.lora.alpha"); ok {
		if value.Type != gguf.ValueTypeFloat32 {
			return nil, errors.New("LoRA alpha is not float32")
		}
		alpha, ok := value.Data.(float32)
		if !ok || math.IsNaN(float64(alpha)) || math.IsInf(float64(alpha), 0) {
			return nil, errors.New("LoRA alpha is invalid")
		}
		result.Alpha = alpha
	}
	if value, ok := file.MetadataValue("adapter.alora.invocation_tokens"); ok {
		if value.Type != gguf.ValueTypeArray || value.ArrayType != gguf.ValueTypeUint32 {
			return nil, errors.New("aLoRA invocation tokens are not uint32 array")
		}
		tokens, ok := value.Data.([]uint32)
		if !ok {
			return nil, errors.New("aLoRA invocation tokens have invalid storage")
		}
		result.InvocationTokens = slices.Clone(tokens)
	}
	baseTensors := make(map[string]gguf.TensorInfo, len(base.Tensors))
	for _, info := range base.Tensors {
		baseTensors[info.Name] = info
	}
	type pair struct {
		a, b *gguf.TensorInfo
	}
	pairs := make(map[string]pair)
	for index := range file.Tensors {
		info := &file.Tensors[index]
		name := info.Name
		var baseName string
		entry := pair{}
		switch {
		case strings.HasSuffix(name, ".lora_a"):
			baseName = strings.TrimSuffix(name, ".lora_a")
			entry = pairs[baseName]
			entry.a = info
		case strings.HasSuffix(name, ".lora_b"):
			baseName = strings.TrimSuffix(name, ".lora_b")
			entry = pairs[baseName]
			entry.b = info
		case strings.HasSuffix(name, "_norm.weight"):
			continue
		default:
			return nil, fmt.Errorf("LoRA tensor %q has unexpected suffix", name)
		}
		if baseName == "" {
			return nil, fmt.Errorf("LoRA tensor %q has empty base name", name)
		}
		pairs[baseName] = entry
	}
	if len(pairs) == 0 {
		return nil, errors.New("LoRA adapter has no tensor pairs")
	}
	for name, pair := range pairs {
		if pair.a == nil || pair.b == nil {
			return nil, fmt.Errorf("LoRA tensor pair for %q is incomplete", name)
		}
		baseInfo, ok := baseTensors[name]
		if !ok {
			return nil, fmt.Errorf("LoRA tensor %q does not exist in base model", name)
		}
		embedding := strings.HasSuffix(name, "token_embd.weight")
		if err := validateLoRAShapes(name, baseInfo, *pair.a, *pair.b, embedding); err != nil {
			return nil, err
		}
		a, err := LoadHostTensor(ctx, file, *pair.a)
		if err != nil {
			return nil, err
		}
		b, err := LoadHostTensor(ctx, file, *pair.b)
		if err != nil {
			return nil, err
		}
		result.Weights[name] = LoRAWeight{A: a, B: b, Embedding: embedding}
	}
	return result, nil
}

func validateLoRAShapes(name string, base, a, b gguf.TensorInfo, embedding bool) error {
	if base.Dimensions < 2 || a.Dimensions != base.Dimensions || b.Dimensions != base.Dimensions {
		return fmt.Errorf("LoRA tensor %q ranks do not match", name)
	}
	if embedding {
		if base.Dimensions != 2 || base.Shape[0] != b.Shape[1] || base.Shape[1] != a.Shape[1] || a.Shape[0] != b.Shape[0] {
			return fmt.Errorf("LoRA embedding tensor %q has incompatible shape", name)
		}
		return nil
	}
	if base.Shape[0] != a.Shape[0] || base.Shape[1] != b.Shape[1] || a.Shape[1] != b.Shape[0] {
		return fmt.Errorf("LoRA tensor %q has incompatible matrix shape", name)
	}
	for axis := uint32(2); axis < base.Dimensions; axis++ {
		if base.Shape[axis] != a.Shape[axis] || base.Shape[axis] != b.Shape[axis] {
			return fmt.Errorf("LoRA tensor %q has incompatible group shape", name)
		}
	}
	return nil
}
