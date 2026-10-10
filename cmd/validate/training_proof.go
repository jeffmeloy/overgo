package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/trainingprogram"
)

// A training proof is the verification its activation accepted, and a
// training activation is verified only by a real run under its recipe
// (Observer.Verify). Each cmd/train route bootstraps its recipe over
// committed data, and the dense workflow trains the recipe it is given over
// that recipe's own data, so the recipe whose objective reads the data is the
// one a run of the route reaches and re-verifies at the commit it executed.
// An objective registered over store data is proved under the selected
// recipe, whose objective must equal the alias's immutable target. A proof
// run, of train -objective for text and of its route for a table, re-verifies
// that recipe, publishing no held-out verdict to supersede the one the
// recipe's quality reads. Each run keeps the bounds the training lane recorded
// for its route.
const (
	trainingProof = "training-proof"
	routeSteps    = "3"
	// The dense workflow takes one short update, lexically frozen so it
	// stays device-resident; an objective's proof run trains the same dense
	// model the same way.
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
	// motGrounding is the stimulus the mot route's recipe is grounded on;
	// the route trains on the golden RxBrain prefill among the reference
	// fixtures.
	motGrounding = "docs/verification/rxbrain-mot-training-stimulus.txt"
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
	{grounding: motGrounding, arguments: []string{"-route", "mot", "-binding", "rxbrain", "-steps", routeSteps, "-max-wall", "30m"}},
	{grounding: hybridGrounding, arguments: []string{"-route", "hybrid-stack", "-lane", "layer-streamed", "-steps", routeSteps, "-max-wall", "120m"},
		environment: []string{"GOGC=50", "GOMEMLIMIT=68GiB"}},
	{grounding: denseGrounding, arguments: []string{
		"-dataset", denseGrounding, "-steps", denseSteps, "-seq", denseSequence, "-max-wall", "10m", "-freeze-lexical",
	}},
}

// objectiveProof bounds the proof run train -objective takes as the dense
// workflow's run is bounded.
var objectiveProof = []string{"-proof", "-steps", denseSteps, "-seq", denseSequence, "-max-wall", "10m", "-freeze-lexical"}

// tabularRoute is the cmd/train route that trains a table-prediction
// objective, and objectiveRoutes names, by the kind of objective, the route
// that trains an objective registered over store data which train -objective,
// training text, does not: its proof run is the route's, under the objective's
// alias and the cell's recipe, bounded as the routes' runs are.
const tabularRoute = "tabular"

var objectiveRoutes = map[trainingprogram.ObjectiveKind]string{
	trainingprogram.ObjectiveTablePrediction: tabularRoute,
}

// groundTraining gives each training cell, in place, the committed data its
// recipe's objective reads when a route grounds its recipe on it, else the
// alias, the first by name, that registers its token-prediction objective,
// and a raw DiT checkpoint the one video stack directory the dit route
// trains from.
func groundTraining(ctx context.Context, root string, store *overgodb.Store, cells []ModelValidation) error {
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
	var registered map[artifact.ID]string
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
			// train -objective trains text; an objective of another kind
			// trains through the route that kind names, where one does.
			route, routed := objectiveRoutes[objective.Kind]
			if cells[index].Grounding == "" && (routed || objective.Kind == trainingprogram.ObjectiveTokenPrediction) {
				if registered == nil {
					if registered, err = aliasNames(ctx, store); err != nil {
						return err
					}
				}
				cells[index].Objective, cells[index].Route = registered[id], route
			}
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

// aliasNames names each target an alias binds by the first alias, in name
// order, that binds it.
func aliasNames(ctx context.Context, store *overgodb.Store) (map[artifact.ID]string, error) {
	names := map[artifact.ID]string{}
	err := store.VisitAliases(ctx, "", func(alias overgodb.AliasView) error {
		if _, named := names[alias.Target]; !named {
			names[alias.Target] = alias.Name
		}
		return nil
	})
	return names, err
}

// trainingRun is the bounded run into store that re-verifies a training cell,
// the environment it runs under, and the scratch directory its checkpoint
// publishes beneath, which -run empties before and after the run: the proof
// is the verification the run records, not the checkpoint. A route trains a
// directory's weights, a GGUF or a raw DiT checkpoint as its own flags name
// them, the dense workflow the recipe the cell names, and an objective's
// proof run the model directory under the objective's alias: through train
// -objective for text, and through its route for a table. It says why
// when no route grounds the cell's recipe and no alias registers its
// objective, or the route lacks an input.
func trainingRun(cell ModelValidation, store string) (args, environment []string, err error) {
	at := slices.IndexFunc(trainingRoutes, func(route trainingRoute) bool { return route.grounding == cell.Grounding })
	route := trainingRoute{arguments: append([]string{"-objective", cell.Objective}, objectiveProof...)}
	if at >= 0 {
		route = trainingRoutes[at]
	} else if cell.Route != "" {
		route.arguments = []string{"-route", cell.Route, "-objective", cell.Objective, "-proof", "-steps", routeSteps}
	}
	inputs := []string{"-model", filepath.Dir(cell.Location)}
	switch {
	case at < 0 && cell.Objective == "":
		return nil, nil, errors.New("no cmd/train route grounds its recipe's objective, and no alias registers it for train -objective")
	case cell.Location == "":
		return nil, nil, errors.New("a training run needs the model's weights")
	case at < 0 && cell.Route == tabularRoute:
		// The catalog finds a table model by one head's weights, in the head's
		// own directory; the route takes the model's, and trains the head the
		// objective's metric names.
		inputs = []string{"-model", filepath.Dir(filepath.Dir(cell.Location)), "-recipe", cell.Recipe.String()}
	case at < 0:
		inputs = append(inputs, "-recipe", cell.Recipe.String())
	case cell.Grounding == motGrounding:
		roots, err := dataroot.ResolveCurrent()
		if err != nil {
			return nil, nil, err
		}
		inputs = append(inputs, "-fixtures", referenceFixtures(roots))
	case cell.Grounding == hybridGrounding:
		inputs = []string{"-model", cell.Location}
	case cell.Grounding == denseGrounding:
		inputs = append(inputs, "-recipe", cell.Recipe.String())
	case cell.Grounding == ditGrounding && filepath.Base(cell.Location) == ditWeights:
		inputs = []string{"-model-dir", filepath.Dir(cell.Location)}
	case cell.Grounding == ditGrounding && cell.Conditioning == "":
		return nil, nil, errors.New("a raw DiT checkpoint trains against a video stack directory no other training activation names")
	case cell.Grounding == ditGrounding:
		inputs = []string{"-model-dir", cell.Conditioning, "-checkpoint", cell.Location}
	}
	args = append([]string{"run", "./cmd/train"}, route.arguments...)
	// The preview names a caller-chosen output; acquisition replaces it with
	// its own private directory before starting the producer.
	args = append(append(args, "-store", store, "-out", "checkpoint"), inputs...)
	return args, route.environment, nil
}
