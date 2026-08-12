package workflowrecipe

import (
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

const (
	ModuleTokenize        recipe.ModuleID = "text.tokenize"
	ModuleDetokenize      recipe.ModuleID = "text.detokenize"
	ModuleGenerate        recipe.ModuleID = "model.generate"
	ModuleEmbed           recipe.ModuleID = "model.embed"
	ModulePool            recipe.ModuleID = "embedding.pool"
	ModuleNormalize       recipe.ModuleID = "embedding.normalize"
	ModuleRerankPrepare   recipe.ModuleID = "rerank.prepare"
	ModuleScore           recipe.ModuleID = "model.score"
	ModuleOrder           recipe.ModuleID = "rerank.order"
	ModuleDecodeImage     recipe.ModuleID = "media.decode-image"
	ModuleDecodeAudio     recipe.ModuleID = "media.decode-audio"
	ModuleDecodeVideo     recipe.ModuleID = "media.decode-video"
	ModuleProject         recipe.ModuleID = "projector.project"
	ModuleBatchDataset    recipe.ModuleID = "training.batch-dataset"
	ModuleTrainingForward recipe.ModuleID = "training.forward"
	ModuleBackward        recipe.ModuleID = "training.backward"
	ModuleOptimize        recipe.ModuleID = "training.optimize"
)

type MediaKind uint8

const (
	MediaImage MediaKind = iota + 1
	MediaAudio
	MediaVideo
)

type Bindings struct {
	Model      artifact.ID
	Profile    artifact.ID
	Tokenizer  artifact.ID
	Projector  artifact.ID
	Dataset    artifact.ID
	Checkpoint artifact.ID
	Adapters   []artifact.ID
}

type Plan struct {
	Program recipe.Program
	Support ExecutionSupport
}

type ExecutionSupport uint8

const (
	ExecutionRuntime ExecutionSupport = iota + 1
	ExecutionOrchestration
)

var catalog = mustCatalog()

func Catalog() *recipe.Catalog {
	return catalog.Clone()
}

func Module(id recipe.ModuleID) (recipe.Module, bool) { return catalog.Module(id) }

func Generation(bindings Bindings, placement recipe.Placement) (recipe.Definition, error) {
	tokenize := node("tokenize", ModuleTokenize, placement)
	generate := node("generate", ModuleGenerate, placement)
	detokenize := node("detokenize", ModuleDetokenize, placement)
	return definition(
		recipe.TaskGeneration, bindings, []recipe.DependencyRole{recipe.DependencyTokenizer},
		[]recipe.Node{tokenize, generate, detokenize},
		[]recipe.Edge{
			edge(tokenize, "tokens", generate, "tokens"),
			edge(generate, "tokens", detokenize, "tokens"),
		},
		[]recipe.Input{input("prompt", recipe.DataText, tokenize, "text")},
		[]recipe.Output{output("text", recipe.DataText, detokenize, "text")},
	)
}

func Embedding(bindings Bindings, placement recipe.Placement) (recipe.Definition, error) {
	tokenize := node("tokenize", ModuleTokenize, placement)
	embed := node("embed", ModuleEmbed, placement)
	pool := node("pool", ModulePool, placement)
	normalize := node("normalize", ModuleNormalize, placement)
	return definition(
		recipe.TaskEmbedding, bindings, []recipe.DependencyRole{recipe.DependencyTokenizer},
		[]recipe.Node{tokenize, embed, pool, normalize},
		[]recipe.Edge{
			edge(tokenize, "tokens", embed, "tokens"),
			edge(embed, "embeddings", pool, "embeddings"),
			edge(pool, "embeddings", normalize, "embeddings"),
		},
		[]recipe.Input{input("text", recipe.DataText, tokenize, "text")},
		[]recipe.Output{output("embeddings", recipe.DataEmbeddings, normalize, "embeddings")},
	)
}

func Rerank(bindings Bindings, placement recipe.Placement) (recipe.Definition, error) {
	prepare := node("prepare", ModuleRerankPrepare, placement)
	score := node("score", ModuleScore, placement)
	order := node("order", ModuleOrder, placement)
	return definition(
		recipe.TaskRerank, bindings, []recipe.DependencyRole{recipe.DependencyTokenizer},
		[]recipe.Node{prepare, score, order},
		[]recipe.Edge{
			edge(prepare, "tokens", score, "tokens"),
			edge(score, "scores", order, "scores"),
		},
		[]recipe.Input{
			input("query", recipe.DataText, prepare, "query"),
			input("documents", recipe.DataText, prepare, "documents"),
		},
		[]recipe.Output{output("ranking", recipe.DataRanking, order, "ranking")},
	)
}

func Projection(bindings Bindings, media MediaKind, placement recipe.Placement) (recipe.Definition, error) {
	module, data, err := mediaContract(media)
	if err != nil {
		return recipe.Definition{}, err
	}
	decode := node("decode", module, placement)
	project := node("project", ModuleProject, placement)
	return definition(
		recipe.TaskProjection, bindings, []recipe.DependencyRole{recipe.DependencyProjector},
		[]recipe.Node{decode, project},
		[]recipe.Edge{edge(decode, "tensor", project, "tensor")},
		[]recipe.Input{input("media", data, decode, "media")},
		[]recipe.Output{output("embeddings", recipe.DataEmbeddings, project, "embeddings")},
	)
}

func Training(bindings Bindings, placement recipe.Placement) (recipe.Definition, error) {
	batch := node("batch", ModuleBatchDataset, placement)
	forward := node("forward", ModuleTrainingForward, placement)
	backward := node("backward", ModuleBackward, placement)
	optimize := node("optimize", ModuleOptimize, placement)
	return definition(
		recipe.TaskTraining, bindings, []recipe.DependencyRole{recipe.DependencyDataset},
		[]recipe.Node{batch, forward, backward, optimize},
		[]recipe.Edge{
			edge(batch, "batch", forward, "batch"),
			edge(forward, "loss", backward, "loss"),
			edge(backward, "gradients", optimize, "gradients"),
		}, nil,
		[]recipe.Output{output("checkpoint", recipe.DataCheckpoint, optimize, "checkpoint")},
	)
}

func Compile(definition recipe.Definition) (Plan, error) {
	if err := definition.Validate(catalog); err != nil {
		return Plan{}, err
	}
	if err := validateTaskDependencies(definition); err != nil {
		return Plan{}, err
	}
	program, err := recipe.CompileProgram(definition, catalog)
	if err != nil {
		return Plan{}, err
	}
	return Plan{
		Program: program, Support: executionSupport(definition.Task),
	}, nil
}

func executionSupport(task recipe.Task) ExecutionSupport {
	if task == recipe.TaskTraining {
		return ExecutionOrchestration
	}
	return ExecutionRuntime
}

func definition(
	task recipe.Task,
	bindings Bindings,
	required []recipe.DependencyRole,
	nodes []recipe.Node,
	edges []recipe.Edge,
	inputs []recipe.Input,
	outputs []recipe.Output,
) (recipe.Definition, error) {
	dependencies, err := bindings.dependencies()
	if err != nil {
		return recipe.Definition{}, err
	}
	for _, role := range required {
		if !hasDependency(dependencies, role, 0) {
			return recipe.Definition{}, fmt.Errorf("workflow recipe: missing %s dependency", role)
		}
	}
	return recipe.NewDefinitionWithDependencies(task, dependencies, nodes, edges, inputs, outputs)
}

func (b Bindings) dependencies() ([]recipe.Dependency, error) {
	dependencies := make([]recipe.Dependency, 0, 6+len(b.Adapters))
	appendID := func(role recipe.DependencyRole, id artifact.ID) {
		if id.Valid() {
			dependencies = append(dependencies, recipe.Dependency{Role: role, Artifact: id})
		}
	}
	appendID(recipe.DependencyModel, b.Model)
	appendID(recipe.DependencyProfile, b.Profile)
	appendID(recipe.DependencyTokenizer, b.Tokenizer)
	appendID(recipe.DependencyProjector, b.Projector)
	appendID(recipe.DependencyDataset, b.Dataset)
	appendID(recipe.DependencyCheckpoint, b.Checkpoint)
	for slot, id := range b.Adapters {
		if !id.Valid() {
			return nil, fmt.Errorf("workflow recipe: invalid adapter slot %d", slot)
		}
		dependencies = append(dependencies, recipe.Dependency{
			Role: recipe.DependencyAdapter, Slot: uint32(slot), Artifact: id,
		})
	}
	return dependencies, nil
}

func validateTaskDependencies(definition recipe.Definition) error {
	required := map[recipe.Task][]recipe.DependencyRole{
		recipe.TaskGeneration: {recipe.DependencyTokenizer},
		recipe.TaskEmbedding:  {recipe.DependencyTokenizer},
		recipe.TaskRerank:     {recipe.DependencyTokenizer},
		recipe.TaskProjection: {recipe.DependencyProjector},
		recipe.TaskTraining:   {recipe.DependencyDataset},
	}[definition.Task]
	if required == nil {
		return fmt.Errorf("workflow recipe: unsupported task %q", definition.Task)
	}
	for _, role := range required {
		if _, ok := definition.Dependency(role, 0); !ok {
			return fmt.Errorf("workflow recipe: missing %s dependency", role)
		}
	}
	return nil
}

func mediaContract(media MediaKind) (recipe.ModuleID, recipe.DataKind, error) {
	switch media {
	case MediaImage:
		return ModuleDecodeImage, recipe.DataImage, nil
	case MediaAudio:
		return ModuleDecodeAudio, recipe.DataAudio, nil
	case MediaVideo:
		return ModuleDecodeVideo, recipe.DataVideo, nil
	default:
		return "", "", errors.New("workflow recipe: invalid media kind")
	}
}

func node(id recipe.NodeID, module recipe.ModuleID, placement recipe.Placement) recipe.Node {
	return recipe.Node{ID: id, Module: module, Placement: placement}
}

func edge(from recipe.Node, fromPort recipe.PortName, to recipe.Node, toPort recipe.PortName) recipe.Edge {
	return recipe.Edge{
		From: recipe.Endpoint{Node: from.ID, Port: fromPort},
		To:   recipe.Endpoint{Node: to.ID, Port: toPort},
	}
}

func input(name recipe.PortName, data recipe.DataKind, target recipe.Node, port recipe.PortName) recipe.Input {
	return recipe.Input{Name: name, Data: data, Target: recipe.Endpoint{Node: target.ID, Port: port}}
}

func output(name recipe.PortName, data recipe.DataKind, source recipe.Node, port recipe.PortName) recipe.Output {
	return recipe.Output{Name: name, Data: data, Source: recipe.Endpoint{Node: source.ID, Port: port}}
}

func hasDependency(dependencies []recipe.Dependency, role recipe.DependencyRole, slot uint32) bool {
	return slices.ContainsFunc(dependencies, func(dependency recipe.Dependency) bool {
		return dependency.Role == role && dependency.Slot == slot
	})
}
