package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/hfconvert"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
)

// trainedModel is a trained model still to be measured: its GGUF, or the directory the view serves, and whether an
// activation names it already, from a run that stopped before its measurement ended.
type trainedModel struct {
	path     string
	admitted bool
}

// admitTrained readies each trained model still to be measured, so its cells can run. A trained model is listed from the
// moment its training ends whether this works or not; one that cannot be readied here, on a tree with changes or with no
// device, is named and taken up by the next run.
func admitTrained(ctx context.Context, output io.Writer, root, repository string) error {
	models, err := unmeasuredTrained(ctx, repository)
	if err != nil {
		return err
	}
	steps, err := newOnboarding(root, repository, output)
	if err != nil {
		return err
	}
	steps.ready(ctx, output, models)
	return nil
}

// ready onboards the models no activation names, and gives each model the benchmark its guard needs. One admitted already
// is not validated again, which would record its goldens over.
func (steps onboarding) ready(ctx context.Context, output io.Writer, models []trainedModel) {
	for _, model := range models {
		var err error
		if !model.admitted {
			_, err = steps.admit(ctx, output, model.path)
		}
		if err == nil {
			err = steps.benchmark(ctx, output, model.path)
		}
		if err != nil {
			fmt.Fprintf(output, "validate: trained model %s is not measured yet, and the next run takes it up: %v\n", model.path, err)
		}
	}
}

// unmeasuredTrained is each trained model present and still to be measured.
func unmeasuredTrained(ctx context.Context, repository string) (models []trainedModel, err error) {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return nil, err
	}
	defer func() { err = errors.Join(err, store.Close()) }()
	entries, truncated, err := discovery.RegisteredCatalog(ctx, store, catalogLimit, discovery.LoadMemo(ctx, store))
	if err != nil {
		return nil, err
	}
	if truncated {
		return nil, fmt.Errorf("validate: registered catalog exceeds listing bound %d; raise the bound", catalogLimit)
	}
	for _, entry := range entries {
		if !entry.Trained || entry.Measured || !entry.Present {
			continue
		}
		path := entry.Location
		if directory := filepath.Dir(path); hfconvert.Viewed(directory) {
			path = directory
		}
		models = append(models, trainedModel{path: path, admitted: len(entry.Capabilities) != 0})
	}
	return models, nil
}

// measuredCells keeps the trained models whose every owed cell was acquired by this run, each with the task and recipe its
// mark names. A model with a cell that failed, or that a run narrowed to one validation did not ask for, has not had its
// measurement, and the next run measures it.
func measuredCells(cells []Selected, acquired map[artifact.ID]int) []ModelValidation {
	var measured []ModelValidation
	seen := map[artifact.ID]bool{}
	for _, sel := range cells {
		if sel.Trained && !sel.Measured && sel.Kind == recipe.TaskInference && !seen[sel.Model] && acquired[sel.Model] == sel.Owed {
			seen[sel.Model] = true
			measured = append(measured, sel.ModelValidation)
		}
	}
	return measured
}

// markMeasured records the one measurement of each model in the store.
func markMeasured(ctx context.Context, storeRoot string, measured []ModelValidation) (err error) {
	if len(measured) == 0 {
		return nil
	}
	repository, err := overgodb.Create(storeRoot)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, repository.Close()) }()
	for _, cell := range measured {
		if err := modelrecipe.MarkMeasured(ctx, repository, cell.Model, cell.Kind, cell.Recipe); err != nil {
			return err
		}
	}
	return nil
}
