package main

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
)

// Focused sample export selects frozen requests, including later executions
// of those same requests, or the exact runs the scorecard's cells are
// measured on. The canonical report retains all recorded media.
type mediaSampleSelection struct {
	// requests are the frozen requests by recipe: a run with the same inputs
	// is an execution of one of them.
	requests map[artifact.ID][][]artifact.ID
	// runs are the attempts the scorecard measures, one run each.
	runs map[artifact.ID]bool
}

func (selection *mediaSampleSelection) includes(run runrecord.Run) bool {
	if selection == nil {
		return true
	}
	if selection.runs != nil {
		return selection.runs[run.ID]
	}
	return slices.ContainsFunc(selection.requests[run.Recipe], func(inputs []artifact.ID) bool {
		return slices.Equal(inputs, run.Inputs)
	})
}

func loadMediaSampleSelection(ctx context.Context, root string, store *overgodb.Store, scope mediaReportScope, entries []discovery.CatalogEntry) (*mediaSampleSelection, error) {
	if scope.MeasuredAttempts {
		return measuredSampleSelection(ctx, store)
	}
	if !scope.FrozenRequests {
		return nil, nil
	}
	var protocol imageVideoProtocol
	if err := jsonfile.DecodeStrict(filepath.Join(root, imageVideoProtocolPath), &protocol); err != nil {
		return nil, err
	}
	if protocol.Version != 1 || len(protocol.Cases) == 0 || len(protocol.RequiredChecks) == 0 {
		return nil, errors.New("media report: incomplete quality protocol")
	}
	selection := &mediaSampleSelection{requests: map[artifact.ID][][]artifact.ID{}}
	seen := map[string]bool{}
	for _, value := range protocol.Cases {
		if value.ID == "" || seen[value.ID] {
			return nil, errors.New("media report: duplicate or missing case identity")
		}
		seen[value.ID] = true
		matches := slices.ContainsFunc(entries, func(entry discovery.CatalogEntry) bool {
			return entry.Model == value.Model && slices.ContainsFunc(entry.Capabilities, func(capability discovery.Capability) bool {
				return capability.Task == value.Task && capability.Recipe == value.Recipe
			})
		})
		if !matches {
			return nil, fmt.Errorf("media report: case %s differs from the selected catalog", value.ID)
		}
		run, err := runrecord.RequireRun(ctx, store, value.Run)
		if err != nil {
			return nil, err
		}
		if err := checkImageVideoCaseRun(value, run); err != nil {
			return nil, err
		}
		selection.requests[value.Recipe] = append(selection.requests[value.Recipe], value.Inputs)
	}
	for _, entry := range entries {
		for _, capability := range entry.Capabilities {
			if len(selection.requests[capability.Recipe]) == 0 {
				return nil, fmt.Errorf("media report: protocol omits %s/%s", entry.Model, capability.Task)
			}
		}
	}
	return selection, nil
}

// measuredSampleSelection selects, for each media activation, the run of the
// attempt the scorecard's first-output cell is measured on, read from that
// cell's source, so the exported sample is the output whose time and memory
// the scorecard reports.
func measuredSampleSelection(ctx context.Context, store *overgodb.Store) (*mediaSampleSelection, error) {
	rows, err := scorecard.Build(ctx, store)
	if err != nil {
		return nil, err
	}
	selection := &mediaSampleSelection{runs: map[artifact.ID]bool{}}
	for _, row := range rows {
		source := row.Cells[metric.FirstOutput].Source
		if !mediaTask(row.Task) || !source.Valid() {
			continue
		}
		observation, err := runrecord.RequireServingObservation(ctx, store, source)
		if err != nil {
			return nil, err
		}
		if observation.Run.Valid() {
			selection.runs[observation.Run] = true
		}
	}
	return selection, nil
}
