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

// admitTrained onboards each trained model no activation names yet, so its cells can run. A trained model is listed from
// the moment its training ends whether this works or not; one that cannot be admitted here, on a tree with changes or
// with no device, is named and taken up by the next run.
func admitTrained(ctx context.Context, output io.Writer, root, repository string) error {
	paths, err := unadmittedTrained(ctx, repository)
	if err != nil {
		return err
	}
	steps, err := newOnboarding(root, repository, output)
	if err != nil {
		return err
	}
	for _, path := range paths {
		if _, err := steps.admit(ctx, output, path); err != nil {
			fmt.Fprintf(output, "validate: trained model %s is not measured yet, and the next run takes it up: %v\n", path, err)
		}
	}
	return nil
}

// unadmittedTrained is where each trained model no activation names is: its GGUF, or the directory the view serves.
func unadmittedTrained(ctx context.Context, repository string) (paths []string, err error) {
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
		if !entry.Trained || entry.Measured || !entry.Present || len(entry.Capabilities) != 0 {
			continue
		}
		path := entry.Location
		if directory := filepath.Dir(path); hfconvert.Viewed(directory) {
			path = directory
		}
		paths = append(paths, path)
	}
	return paths, nil
}

// measuredCells keeps the trained models whose every cell of this run was acquired, each with the task and recipe its
// mark names; a model with a cell that failed has not had its measurement, and the next run measures it.
func measuredCells(cells []Selected, acquired map[artifact.ID]int) []ModelValidation {
	counted := map[artifact.ID]int{}
	for _, sel := range cells {
		counted[sel.Model]++
	}
	var measured []ModelValidation
	seen := map[artifact.ID]bool{}
	for _, sel := range cells {
		if sel.Trained && !sel.Measured && sel.Kind == recipe.TaskInference && !seen[sel.Model] && acquired[sel.Model] == counted[sel.Model] {
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
