package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/longform"
	"overgo/internal/model"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
)

// Exact model identity remains the conservative coverage class until an
// operator-specific equivalence proof permits substituting another model.
// Storage labels alone never establish geometry, cache or modality coverage.
type guardCoverageEntry struct {
	Model          artifact.ID    `json:"model"`
	Weights        artifact.ID    `json:"weights"`
	Recipe         artifact.ID    `json:"recipe"`
	Definition     artifact.ID    `json:"definition,omitzero"`
	Profile        artifact.ID    `json:"profile,omitzero"`
	Location       string         `json:"location"`
	Task           recipe.Task    `json:"task"`
	Nodes          []recipe.Node  `json:"nodes"`
	Spec           model.Spec     `json:"spec"`
	TensorStorage  map[string]int `json:"tensor_storage"`
	Baseline       string         `json:"baseline"`
	MeasuredWallNS int64          `json:"measured_wall_ns"`
	Gap            string         `json:"gap"`
}

type guardCoverageReport struct {
	Surface    string               `json:"surface"`
	Selected   int                  `json:"selected_models"`
	Explicit   int                  `json:"explicit_records"`
	Covered    int                  `json:"covered_models"`
	Uncovered  int                  `json:"uncovered_models"`
	Unused     []string             `json:"unused_records"`
	Entries    []guardCoverageEntry `json:"entries"`
	Exclusions string               `json:"exclusions"`
}

type resolveGuardFacts func(target) (recipe.Definition, modelrecipe.ResolvedModelDefinition, error)

func reportGuardCoverage(ctx context.Context, output io.Writer, options options, targets []target, surface string) error {
	baselines, err := readBaselines(ctx, options)
	if err != nil {
		return err
	}
	store, err := overgodb.OpenReadOnly(options.Repository)
	if err != nil {
		return err
	}
	defer store.Close()
	report, err := inspectGuardCoverage(targets, baselines, surface, func(target target) (recipe.Definition, modelrecipe.ResolvedModelDefinition, error) {
		active, found, err := modelrecipe.ActiveRecord(ctx, store, target.entry.Model, recipe.TaskInference)
		if err != nil || !found || active.Definition.ID != target.entry.Recipe {
			return recipe.Definition{}, modelrecipe.ResolvedModelDefinition{}, errors.Join(errors.New("active recipe changed or is absent"), err)
		}
		id, found := active.Definition.PrimaryDependency(recipe.DependencyDefinition)
		if !found {
			return active.Definition, modelrecipe.ResolvedModelDefinition{}, errors.New("active recipe has no model definition")
		}
		resolved, err := modelrecipe.ResolveModelDefinition(ctx, store, id)
		return active.Definition, resolved, err
	})
	if writeErr := clioptions.WritePrettyJSON(output, report); writeErr != nil {
		return errors.Join(err, writeErr)
	}
	return err
}

func inspectGuardCoverage(targets []target, baselines map[artifact.ID]longform.Summary, surface string, resolve resolveGuardFacts) (guardCoverageReport, error) {
	report := guardCoverageReport{Surface: surface, Selected: len(targets), Explicit: len(baselines),
		Exclusions: "selected servable text models only; exact model coverage, no cross-model substitution; no models loaded, fresh measurements, GPU tests, benchmark suites or modality validation"}
	if len(targets) == 0 || surface == "" {
		return report, errors.New("guard coverage: selected models and source surface are required")
	}
	used := map[artifact.ID]bool{}
	for _, target := range targets {
		entry := guardCoverageEntry{Model: target.entry.Model, Weights: target.weights, Recipe: target.entry.Recipe, Location: target.entry.Location}
		definition, resolved, err := resolve(target)
		if err == nil {
			entry.Definition, entry.Profile = resolved.Document.ID, resolved.Document.Profile
			entry.Task, entry.Nodes, entry.Spec = definition.Task, slices.Clone(definition.Nodes), resolved.Spec
			entry.TensorStorage = map[string]int{}
			for _, tensor := range resolved.Tensors.Tensors {
				entry.TensorStorage[tensor.Storage]++
			}
			id, found := definition.PrimaryDependency(recipe.DependencyDefinition)
			if !found || id != entry.Definition || definition.ID != entry.Recipe || resolved.Document.Model != entry.Model ||
				len(entry.TensorStorage) == 0 || len(entry.Nodes) == 0 {
				err = errors.New("execution facts do not bind the selected model and active recipe")
			}
		}
		baseline, found := baselines[target.weights]
		if found {
			used[target.weights] = true
			entry.Baseline, entry.MeasuredWallNS = baseline.Record.String(), baseline.Result.WallNS
		}
		if err == nil {
			switch {
			case !found:
				err = errors.New("no explicit accepted baseline")
			case baseline.Result.Surface != surface:
				err = errors.New("baseline source surface is stale")
			case baseline.Result.Program.Model != entry.Model || baseline.Result.Program.Recipe != entry.Recipe ||
				baseline.Result.Program.Definition != entry.Definition || baseline.Result.Program.Profile != entry.Profile:
				err = errors.New("baseline execution identity differs from the active recipe")
			default:
				err = validateGuard(baseline.Result)
			}
		}
		if err != nil {
			entry.Gap = err.Error()
			report.Uncovered++
		} else {
			report.Covered++
		}
		report.Entries = append(report.Entries, entry)
	}
	for model, baseline := range baselines {
		if !used[model] {
			report.Unused = append(report.Unused, baseline.Record.String())
		}
	}
	slices.Sort(report.Unused)
	slices.SortFunc(report.Entries, func(a, b guardCoverageEntry) int { return strings.Compare(a.Model.String(), b.Model.String()) })
	if report.Uncovered != 0 || len(report.Unused) != 0 {
		return report, fmt.Errorf("guard coverage: covered=%d/%d uncovered=%d unused_records=%d; incomplete, not passing evidence", report.Covered, report.Selected, report.Uncovered, len(report.Unused))
	}
	return report, nil
}
