package main

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"net/url"
	"path"
	"slices"
	"strings"

	"overgo/internal/modelartifact"
)

// sampler is the node class whose sigmas a recording keeps: KSampler, which computes its sigmas from its model's
// sampling, its scheduler, its steps and its denoise.
const sampler = "KSampler"

// sigmaNodes adds to request, beside each KSampler, the nodes that give the sigmas it samples by as text: ComfyUI's
// BasicScheduler, fed the sampler's own model, scheduler, steps and denoise, which computes them as KSampler does;
// KJNodes' SigmasToFloat, a list of their float32 values; and a PreviewAny, whose text is that list as JSON. It returns
// the PreviewAny's id by sampler.
func sigmaNodes(request map[string]apiNode) map[string]string {
	previews := map[string]string{}
	for _, id := range slices.Sorted(maps.Keys(request)) {
		node := request[id]
		if node.ClassType != sampler {
			continue
		}
		scheduler, floats, preview := "overgo-sigmas-"+id, "overgo-sigma-floats-"+id, "overgo-sigma-text-"+id
		inputs := map[string]any{}
		for _, name := range []string{"model", "scheduler", "steps", "denoise"} {
			inputs[name] = node.Inputs[name]
		}
		request[scheduler] = apiNode{ClassType: "BasicScheduler", Inputs: inputs}
		request[floats] = apiNode{ClassType: "SigmasToFloat", Inputs: map[string]any{"sigmas": []any{scheduler, 0}}}
		request[preview] = apiNode{ClassType: "PreviewAny", Inputs: map[string]any{"source": []any{floats, 0}}}
		previews[id] = preview
	}
	return previews
}

// readSigmas is the sigmas a sigma preview's text holds, refused unless it is a JSON list of more than one value.
func readSigmas(texts []string) ([]float64, error) {
	if len(texts) != 1 {
		return nil, fmt.Errorf("comfy-reference: a sigma preview gave %d texts; want one", len(texts))
	}
	var sigmas []float64
	if err := json.Unmarshal([]byte(texts[0]), &sigmas); err != nil || len(sigmas) < 2 {
		return nil, fmt.Errorf("comfy-reference: a sigma preview gave %q; want a JSON list of the sigmas (%v)", texts[0], err)
	}
	return sigmas, nil
}

// recordedModel is a model file the posted request names, at a node's input (a nested input by its path, "lora_2.lora"),
// as ComfyUI lists it in its models folder, and the SHA-256 the import recipe pins it at.
type recordedModel struct {
	Node   string `json:"node"`
	Input  string `json:"input"`
	Folder string `json:"folder"`
	Name   string `json:"name"`
	SHA256 string `json:"sha256"`
}

// models are the model files of recipe the request names, each read from the install's folder ComfyUI lists it in
// (/models/<folder>), the folder the recipe's source path in the install names. A file named in a nested input that is
// switched off ("on": false, as rgthree's Power Lora Loader writes one) is not loaded and is not named; a file ComfyUI
// does not list is refused.
func (r recorder) models(ctx context.Context, recipe modelartifact.LocalImport, request map[string]apiNode) ([]recordedModel, error) {
	type pinned struct{ folder, sha256 string }
	byName := map[string]pinned{}
	for _, file := range recipe.Files {
		folder, name := path.Split(strings.TrimPrefix(file.Source, "ComfyUI/models/"))
		if folder == "" || !strings.HasPrefix(file.Source, "ComfyUI/models/") {
			return nil, fmt.Errorf("comfy-reference: the recipe's %s comes from %s, not a folder of ComfyUI's models", file.Path, file.Source)
		}
		byName[name] = pinned{strings.TrimSuffix(folder, "/"), file.SHA256}
	}
	var named []recordedModel
	var walk func(node, input string, value any)
	walk = func(node, input string, value any) {
		switch value := value.(type) {
		case string:
			if file, held := byName[value]; held {
				named = append(named, recordedModel{Node: node, Input: input, Folder: file.folder, Name: value, SHA256: file.sha256})
			}
		case map[string]any:
			if on, given := value["on"].(bool); given && !on {
				return
			}
			for _, key := range slices.Sorted(maps.Keys(value)) {
				walk(node, input+"."+key, value[key])
			}
		}
	}
	for _, id := range slices.Sorted(maps.Keys(request)) {
		for _, input := range slices.Sorted(maps.Keys(request[id].Inputs)) {
			walk(id, input, request[id].Inputs[input])
		}
	}
	listed := map[string][]string{}
	for _, model := range named {
		if _, read := listed[model.Folder]; !read {
			var names []string
			if err := r.call(ctx, http.MethodGet, "/models/"+url.PathEscape(model.Folder), nil, &names); err != nil {
				return nil, err
			}
			listed[model.Folder] = names
		}
		if !slices.Contains(listed[model.Folder], model.Name) {
			return nil, fmt.Errorf("comfy-reference: the request names %s at %s.%s, which ComfyUI's %s does not list", model.Name, model.Node, model.Input, model.Folder)
		}
	}
	return named, nil
}
