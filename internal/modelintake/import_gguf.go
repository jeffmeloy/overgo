package modelintake

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/hfgguf"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
)

func inspectGGUFImport(ctx context.Context, path string, output hfgguf.LanguageOutput) (modelartifact.Inventory, importComponent, modelrecipe.ProfileDocument, error) {
	component := importComponent{Declaration: modelartifact.PipelineComponent{Name: "."}}
	file, err := gguf.Open(path)
	if err != nil {
		return modelartifact.Inventory{}, component, modelrecipe.ProfileDocument{}, err
	}
	defer file.Close()
	inventory, err := modelartifact.FromGGUFContext(ctx, file, artifact.KindModel)
	if err != nil {
		return inventory, component, modelrecipe.ProfileDocument{}, err
	}
	if err := ctx.Err(); err != nil {
		return inventory, component, modelrecipe.ProfileDocument{}, err
	}
	spec, err := model.ReadSpec(file)
	if unsupported, ok := errors.AsType[*model.UnsupportedArchitectureError](err); ok {
		component.Unsupported = unsupported.Error()
		return inventory, component, modelrecipe.ProfileDocument{}, nil
	}
	if err != nil {
		return inventory, component, modelrecipe.ProfileDocument{}, err
	}
	weights, err := model.ReadWeights(file, spec)
	if err != nil {
		return inventory, component, modelrecipe.ProfileDocument{}, err
	}
	policy := importPolicy{Source: inventory.Manifest.ID, Name: "tokenizer", Role: artifact.ComponentTokenizer, Parameters: map[string]json.RawMessage{}}
	for _, entry := range file.Metadata {
		if !strings.HasPrefix(entry.Key, "tokenizer.") {
			continue
		}
		if entry.Value.Type == gguf.ValueTypeArray {
			continue
		}
		encoded, err := json.Marshal(entry.Value.Data)
		if err != nil {
			return inventory, component, modelrecipe.ProfileDocument{}, err
		}
		policy.Parameters[entry.Key] = encoded
	}
	if len(policy.Parameters) != 0 {
		component.Policies = []importPolicy{policy}
	}
	if output == "" {
		output = hfgguf.LanguageTokens
	}
	if output == hfgguf.LanguageHidden && spec.NextNPredictLayers != 0 {
		component.Unsupported = "GGUF hidden-state import with prediction layers has no registered output mapping"
		return inventory, component, modelrecipe.ProfileDocument{}, nil
	}
	bindings := make([]hfgguf.TensorBinding, 0, len(file.Tensors))
	for _, info := range file.Tensors {
		if output == hfgguf.LanguageHidden && (info.Name == "output.weight" || info.Name == "output.bias") {
			continue
		}
		shape := gguf.ReverseShape(info.Extents())
		bindings = append(bindings, hfgguf.TensorBinding{Source: info.Name, Target: info.Name, SourceShape: shape, TargetShape: slices.Clone(shape), SourceStorage: info.Type.String(), TargetStorage: info.Type.String(), SourceBytes: info.Size, TargetBytes: info.Size})
	}
	slices.SortFunc(bindings, func(a, b hfgguf.TensorBinding) int { return strings.Compare(a.Source, b.Source) })
	if output == hfgguf.LanguageHidden {
		weights.Output = nil
		weights.OutputBias = nil
	}
	component, profile, err := describeImportLanguage(component, spec, weights, bindings, output)
	return inventory, component, profile, err
}
