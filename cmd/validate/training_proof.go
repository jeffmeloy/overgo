package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/trainingprogram"
)

// A training proof is the verification its activation accepted, and a
// training activation is verified only by a real run under its recipe
// (Observer.Verify). Each cmd/train route bootstraps its recipe over
// committed data, and the dense workflow trains the recipe it is given over
// that recipe's own data, so the recipe whose objective reads the data is the
// one a run of the route reaches and re-verifies at the commit it executed.
// Each run keeps the bounds the training lane recorded for its route.
const (
	trainingProof = "training-proof"
	routeSteps    = "3"
	// The dense workflow takes one short update, lexically frozen so it
	// stays device-resident.
	denseSteps    = "1"
	denseSequence = "128"
	// ditWeights is the file the dit route trains from a video stack
	// directory; any other weights it trains are a raw checkpoint that
	// conditions on such a stack.
	ditWeights      = "diffusion_pytorch_model.safetensors"
	ditGrounding    = "docs/verification/wan-dit-training-stimulus.txt"
	hybridGrounding = "fixtures/qwen35_4b_serving_golden.json"
	// denseGrounding is the training lane's smoke corpus, byte for byte: the
	// data its dense-workflow recipes were bootstrapped over.
	denseGrounding = "docs/verification/dense-training-stimulus.txt"
	// motFixtures holds the golden RxBrain prefill the mot route trains on,
	// beside the models it was captured from.
	motFixtures = "C:/Users/jeffm/adaptive_new/fixtures"
)

// trainingRoute is one route a training proof runs: the committed data its
// recipe's objective reads, cmd/train's arguments that select and bound it,
// and the environment its runs were recorded under.
type trainingRoute struct {
	grounding   string
	arguments   []string
	environment []string
}

var trainingRoutes = []trainingRoute{
	{grounding: ditGrounding, arguments: []string{"-route", "dit", "-schedule-index", "0", "-steps", routeSteps, "-max-wall", "60m"}},
	{grounding: "docs/verification/rxbrain-mot-training-stimulus.txt", arguments: []string{
		"-route", "mot", "-fixtures", motFixtures, "-binding", "rxbrain", "-steps", routeSteps, "-max-wall", "30m",
	}},
	{grounding: hybridGrounding, arguments: []string{"-route", "hybrid-stack", "-lane", "layer-streamed", "-steps", routeSteps, "-max-wall", "120m"},
		environment: []string{"GOGC=50", "GOMEMLIMIT=68GiB"}},
	{grounding: denseGrounding, arguments: []string{
		"-dataset", denseGrounding, "-steps", denseSteps, "-seq", denseSequence, "-max-wall", "10m", "-freeze-lexical",
	}},
}

// groundTraining gives each training cell, in place, the committed data its
// recipe's objective reads when a route grounds its recipe on it, and a raw
// DiT checkpoint the one video stack directory the dit route trains from.
func groundTraining(ctx context.Context, root string, store artifact.Reader, cells []ModelValidation) error {
	grounded := map[artifact.ID]string{}
	for _, route := range trainingRoutes {
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(route.grounding)))
		if err != nil {
			return err
		}
		id, err := artifact.IdentifyBytes(artifact.KindDataset, data)
		if err != nil {
			return err
		}
		grounded[id] = route.grounding
	}
	var stacks []string
	for index, cell := range cells {
		if cell.Validation != trainingProof || !cell.Recipe.Valid() {
			continue
		}
		definition, err := recipe.RequireDefinition(ctx, store, cell.Recipe)
		if err != nil {
			return err
		}
		if id, declared := definition.PrimaryDependency(recipe.DependencyObjective); declared {
			objective, err := trainingprogram.LoadObjective(ctx, store, id)
			if err != nil {
				return err
			}
			cells[index].Grounding = grounded[objective.Dataset]
		}
		if cells[index].Grounding == ditGrounding && filepath.Base(cell.Location) == ditWeights {
			stacks = append(stacks, filepath.Dir(cell.Location))
		}
	}
	slices.Sort(stacks)
	if stacks = slices.Compact(stacks); len(stacks) != 1 {
		return nil
	}
	for index, cell := range cells {
		if cell.Grounding == ditGrounding && filepath.Base(cell.Location) != ditWeights {
			cells[index].Conditioning = stacks[0]
		}
	}
	return nil
}

// trainingRun is the bounded run into store that re-verifies a training cell,
// the environment it runs under, and the scratch directory its checkpoint
// publishes beneath, which -run empties before and after the run: the proof
// is the verification the run records, not the checkpoint. A route trains a
// directory's weights, a GGUF or a raw DiT checkpoint as its own flags name
// them, and the dense workflow the recipe the cell names. It says why when no
// route grounds the cell's recipe or the route lacks an input.
func trainingRun(cell ModelValidation, store string) (args, environment []string, scratch string, err error) {
	at := slices.IndexFunc(trainingRoutes, func(route trainingRoute) bool { return route.grounding == cell.Grounding })
	inputs := []string{"-model", filepath.Dir(cell.Location)}
	switch {
	case at < 0:
		return nil, nil, "", errors.New("no cmd/train route grounds its recipe's objective")
	case cell.Location == "":
		return nil, nil, "", errors.New("a training run needs the model's weights")
	case cell.Grounding == hybridGrounding:
		inputs = []string{"-model", cell.Location}
	case cell.Grounding == denseGrounding:
		inputs = append(inputs, "-recipe", cell.Recipe.String())
	case cell.Grounding == ditGrounding && filepath.Base(cell.Location) == ditWeights:
		inputs = []string{"-model-dir", filepath.Dir(cell.Location)}
	case cell.Grounding == ditGrounding && cell.Conditioning == "":
		return nil, nil, "", errors.New("a raw DiT checkpoint trains against a video stack directory no other training activation names")
	case cell.Grounding == ditGrounding:
		inputs = []string{"-model-dir", cell.Conditioning, "-checkpoint", cell.Location}
	}
	scratch = filepath.Join(os.TempDir(), "overgo-validate-proof", cell.Recipe.DigestHex())
	args = append([]string{"run", "./cmd/train"}, trainingRoutes[at].arguments...)
	args = append(append(args, "-store", store, "-out", filepath.Join(scratch, "checkpoint")), inputs...)
	return args, trainingRoutes[at].environment, scratch, nil
}
