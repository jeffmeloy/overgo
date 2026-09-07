package mediacapability

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/modelartifact"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/seq2seq"
	"overgo/internal/seriesforecast"
	"overgo/internal/speechsynth"
	"overgo/internal/tabularicl"
	"overgo/internal/textgeneration"
	"overgo/internal/thoughtbank"
)

// Capability is one task the catalog can register and execute: Resolve reads a
// model path into its inventory and recipe definition; Execute runs the
// compiled program against the model bytes with a JSON input.
type Capability struct {
	Resolve func(string) (Source, error)
	Execute capabilityruntime.Executor
}

// Executor reports the capability's executor, so a consumer binding a
// catalog by task names the executor without naming this type.
func (capability Capability) Executor() capabilityruntime.Executor { return capability.Execute }

// Source is what Resolve reads from a model path: the model inventory, the
// related inventories (a projector), the component-group manifests, and the
// definition the recipe is built from.
type Source struct {
	Inventory modelartifact.Inventory
	Related   []modelartifact.Inventory
	// manifests are component-group model manifests the source derives
	// from the composite inventory, published so slotted recipe stages
	// resolve their component models by content identity.
	Manifests []artifact.Manifest
	Define    func(artifact.ID) (recipe.Definition, []artifact.Content, error)
}

func definitionSource(
	inventory modelartifact.Inventory,
	err error,
	define func(artifact.ID) (recipe.Definition, error),
) (Source, error) {
	if err != nil {
		return Source{}, err
	}
	return Source{Inventory: inventory, Define: func(modelID artifact.ID) (recipe.Definition, []artifact.Content, error) {
		definition, err := define(modelID)
		return definition, nil, err
	}}, nil
}

func inventoryCapability(
	inventory func(string) (modelartifact.Inventory, error),
	execute capabilityruntime.Executor,
) Capability {
	return Capability{
		Resolve: func(path string) (Source, error) {
			resolved, err := inventory(path)
			return Source{Inventory: resolved}, err
		},
		Execute: execute,
	}
}

func sessionExecutor[Input, Model, Output any](
	director *capabilityruntime.ModelSessionDirector[Input, Model, Output],
	err error,
) capabilityruntime.Executor {
	if err == nil {
		return director.Executor()
	}
	return func(context.Context, artifact.Repository, string, modelrecipe.CapabilityEvidenceSelection, string) (any, error) {
		return nil, err
	}
}

func thoughtBankCapability() Capability {
	director, err := capabilityruntime.NewModelSessionDirector[textgeneration.Request, *thoughtbank.Generator, thoughtbank.Generation](
		"generation", "host", recipe.SessionRequestCapacity,
		textgeneration.Validate,
		func(_ context.Context, _ artifact.Repository, path string, _ recipe.Program, _ textgeneration.Request) (*thoughtbank.Generator, error) {
			return thoughtbank.LoadGenerator(path)
		},
		func(context.Context, *thoughtbank.Generator, textgeneration.Request) error { return nil },
		thoughtbank.RegisterRuntime,
	)
	return Capability{
		Resolve: func(path string) (Source, error) {
			inventory, inventoryErr := thoughtBankInventory(path)
			return definitionSource(inventory, inventoryErr, func(modelID artifact.ID) (recipe.Definition, error) {
				return modelrecipe.CapabilityDefinition(recipe.TaskGeneration, modelID)
			})
		},
		Execute: sessionExecutor(director, err),
	}
}

// Projection is the projection capability for one projector file: the model
// and the projector bind into a projection recipe (no executor).
func Projection(ctx context.Context, store overgodb.DocumentReader, projectorPath string) Capability {
	return Capability{Resolve: func(modelPath string) (Source, error) {
		candidate, err := modelintake.PrepareProjectionCandidate(ctx, store, modelPath, projectorPath)
		if err != nil {
			return Source{}, err
		}
		return Source{Inventory: candidate.Model, Related: []modelartifact.Inventory{candidate.Projector},
			Define: func(modelID artifact.ID) (recipe.Definition, []artifact.Content, error) {
				if modelID != candidate.Model.Manifest.ID {
					return recipe.Definition{}, nil, errors.New("projection model identity changed")
				}
				var contents []artifact.Content
				if candidate.Processor != nil {
					content, err := candidate.Processor.Content()
					if err != nil {
						return recipe.Definition{}, nil, err
					}
					contents = append(contents, content)
				}
				if candidate.Config != nil {
					content, err := candidate.Config.Content()
					if err != nil {
						return recipe.Definition{}, nil, err
					}
					contents = append(contents, content)
				}
				return candidate.Definition, contents, nil
			},
		}, nil
	}}
}

// Catalog binds every registrable task to its capability; cmd/recipe and the
// server generation workspace execute the same code through it.
var Catalog = map[recipe.Task]Capability{
	recipe.TaskGeneration: thoughtBankCapability(),
	recipe.TaskForecast: inventoryCapability(modelartifact.FromHFPath, capabilityruntime.JSONScalar[[]float32, *seriesforecast.Model, []float32](
		"forecast", seriesforecast.ValidateRequest,
		capabilityruntime.IgnoreInput[[]float32](seriesforecast.Load), seriesforecast.RegisterRuntime)),
	recipe.TaskTabular: inventoryCapability(tabularInventory, capabilityruntime.JSONScalar[tabularicl.Request, *tabularicl.Model, tabularicl.Prediction](
		"tabular", tabularicl.ValidateRequest,
		func(_ context.Context, _ artifact.Repository, path string, _ recipe.Program, request tabularicl.Request) (*tabularicl.Model, error) {
			return tabularicl.LoadTask(path, request.Task)
		}, tabularicl.RegisterRuntime)),
	recipe.TaskSeq2Seq: inventoryCapability(modelartifact.FromHFPath, capabilityruntime.JSONScalar[textgeneration.Request, *seq2seq.Generator, string](
		"seq2seq", textgeneration.Validate,
		capabilityruntime.IgnoreInput[textgeneration.Request](seq2seq.LoadGenerator), seq2seq.RegisterRuntime)),
	recipe.TaskSpeech: inventoryCapability(speechInventory, capabilityruntime.JSONScalar[speechsynth.SynthesisRequest, *speechsynth.Synthesizer, speechsynth.Audio](
		"speech", speechsynth.ValidateSynthesisRequest,
		capabilityruntime.IgnoreInput[speechsynth.SynthesisRequest](speechsynth.LoadSynthesizer), speechsynth.RegisterRuntime)),
	recipe.TaskImageGen: imageCapability(),
	recipe.TaskVideoGen: videoCapability(),
	recipe.TaskVQA:      vqaCapability(),
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
			return modelartifact.Inventory{}, fmt.Errorf("%s Inventory: multiple safetensors files in %s", context, path)
		}
		weights = entry.Name()
	}
	if weights == "" {
		return modelartifact.Inventory{}, fmt.Errorf("%s Inventory: no safetensors weights in %s", context, path)
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

func thoughtBankInventory(path string) (modelartifact.Inventory, error) {
	return modelartifact.FromFiles(path, []modelartifact.FileSpec{
		{Path: filepath.Join(path, "model.pt"), Name: "weights", Role: artifact.ComponentWeights},
		{Path: filepath.Join(path, "tokenizer.json"), Name: "tokenizer", Role: artifact.ComponentTokenizer},
		{Path: filepath.Join(path, "generation_config.json"), Name: "generation", Role: artifact.ComponentConfig},
	})
}
