package capabilityruntime

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/hfrepo"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/oscillatorimage"
	"overgo/internal/recipe"
	"overgo/internal/seq2seq"
	"overgo/internal/seriesforecast"
	"overgo/internal/speechsynth"
	"overgo/internal/tabularicl"
)

type Inventory func(string) (modelartifact.Inventory, error)

type Capability struct {
	Inventory Inventory
	Execute   Executor
}

func executable(inventory Inventory, execute Executor) Capability {
	return Capability{Inventory: inventory, Execute: execute}
}

var catalog = map[recipe.Task]Capability{
	recipe.TaskForecast: executable(hfInventory,
		JSONScalar[[]float32, *seriesforecast.Model, []float32](
			"forecast", seriesforecast.ValidateRequest,
			IgnoreInput[[]float32](seriesforecast.Load), seriesforecast.RegisterRuntime)),
	recipe.TaskTabular: executable(tabularInventory,
		JSONScalar[tabularicl.Request, *tabularicl.Model, tabularicl.Prediction](
			"tabular", tabularicl.ValidateRequest,
			func(path string, request tabularicl.Request) (*tabularicl.Model, error) {
				return tabularicl.LoadTask(path, request.Task)
			}, tabularicl.RegisterRuntime)),
	recipe.TaskSeq2Seq: executable(hfInventory,
		JSONScalar[seq2seq.GenerateRequest, *seq2seq.Model, []int](
			"seq2seq", seq2seq.ValidateGenerateRequest,
			IgnoreInput[seq2seq.GenerateRequest](seq2seq.Load), seq2seq.RegisterRuntime)),
	recipe.TaskSpeech: executable(speechInventory,
		JSONScalar[speechsynth.SynthesisRequest, *speechsynth.Synthesizer, speechsynth.Audio](
			"speech", speechsynth.ValidateSynthesisRequest,
			IgnoreInput[speechsynth.SynthesisRequest](speechsynth.LoadSynthesizer), speechsynth.RegisterRuntime)),
	recipe.TaskImageGen: executable(imageGenInventory,
		JSONScalar[oscillatorimage.Request, *oscillatorimage.Model, oscillatorimage.Image](
			"image-gen", oscillatorimage.ValidateRequest,
			IgnoreInput[oscillatorimage.Request](oscillatorimage.Load), oscillatorimage.RegisterRuntime)),
	recipe.TaskVQA: {Inventory: hfInventory},
}

func Lookup(task recipe.Task) (Capability, bool) {
	capability, ok := catalog[task]
	return capability, ok
}

func ResolveActive(
	ctx context.Context,
	store artifact.Reader,
	path string,
	task recipe.Task,
) (modelartifact.Inventory, recipe.Program, error) {
	capability, ok := Lookup(task)
	if !ok {
		return modelartifact.Inventory{}, recipe.Program{}, fmt.Errorf("capability runtime: unsupported task %q", task)
	}
	inventory, err := capability.Inventory(path)
	if err != nil {
		return modelartifact.Inventory{}, recipe.Program{}, err
	}
	_, program, err := modelrecipe.ResolveActiveCapability(ctx, store, inventory.Manifest.ID, task)
	return inventory, program, err
}

func hfInventory(path string) (modelartifact.Inventory, error) {
	repository, err := hfrepo.Open(path)
	if err != nil {
		return modelartifact.Inventory{}, err
	}
	defer repository.Close()
	return modelartifact.FromHFRepository(repository)
}

func safetensorsInventory(context, path, config string, companions ...modelartifact.FileSpec) (modelartifact.Inventory, error) {
	entries, err := os.ReadDir(path)
	if err != nil {
		return modelartifact.Inventory{}, err
	}
	weights := ""
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(strings.ToLower(entry.Name()), ".safetensors") {
			continue
		}
		if weights != "" {
			return modelartifact.Inventory{}, fmt.Errorf("%s inventory: multiple safetensors files in %s", context, path)
		}
		weights = entry.Name()
	}
	if weights == "" {
		return modelartifact.Inventory{}, fmt.Errorf("%s inventory: no safetensors weights in %s", context, path)
	}
	specs := []modelartifact.FileSpec{
		{Path: filepath.Join(path, config), Name: "config", Role: artifact.ComponentConfig},
		{Path: filepath.Join(path, weights), Name: "weights", Role: artifact.ComponentWeights},
	}
	for _, companion := range companions {
		companion.Path = filepath.Join(path, companion.Path)
		specs = append(specs, companion)
	}
	return modelartifact.FromFiles(path, specs)
}

func speechInventory(path string) (modelartifact.Inventory, error) {
	return safetensorsInventory("speech", path, "pockettts_config.json",
		modelartifact.FileSpec{Path: "tokenizer.model", Name: "tokenizer", Role: artifact.ComponentTokenizer})
}

func imageGenInventory(path string) (modelartifact.Inventory, error) {
	return safetensorsInventory("image-gen", path, "config.json")
}

func tabularInventory(path string) (modelartifact.Inventory, error) {
	var specs []modelartifact.FileSpec
	for _, head := range tabularicl.Tasks() {
		specs = append(specs,
			modelartifact.FileSpec{Path: filepath.Join(path, head, "config.json"), Name: head + "/config", Role: artifact.ComponentConfig},
			modelartifact.FileSpec{Path: filepath.Join(path, head, "model.safetensors"), Name: head + "/weights", Role: artifact.ComponentWeights},
		)
	}
	return modelartifact.FromFiles(path, specs)
}
