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
)

// A focused report selects the frozen requests, including later executions of
// those same requests. The all-media report retains its full recorded scope.
type mediaSampleSelection map[artifact.ID][][]artifact.ID

func (selection mediaSampleSelection) includes(run runrecord.Run) bool {
	if selection == nil {
		return true
	}
	return slices.ContainsFunc(selection[run.Recipe], func(inputs []artifact.ID) bool {
		return slices.Equal(inputs, run.Inputs)
	})
}

func loadMediaSampleSelection(ctx context.Context, root string, store *overgodb.Store, scope mediaReportScope, entries []discovery.CatalogEntry) (mediaSampleSelection, error) {
	if scope.Path != imageVideoReportPath {
		return nil, nil
	}
	var protocol imageVideoProtocol
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs", "image_video_protocol.json"), &protocol); err != nil {
		return nil, err
	}
	if protocol.Version != 1 || len(protocol.Cases) == 0 || len(protocol.RequiredChecks) == 0 {
		return nil, errors.New("media report: incomplete quality protocol")
	}
	selection := mediaSampleSelection{}
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
		selection[value.Recipe] = append(selection[value.Recipe], value.Inputs)
	}
	for _, entry := range entries {
		for _, capability := range entry.Capabilities {
			if len(selection[capability.Recipe]) == 0 {
				return nil, fmt.Errorf("media report: protocol omits %s/%s", entry.Model, capability.Task)
			}
		}
	}
	return selection, nil
}
