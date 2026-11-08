package main

import (
	"cmp"
	"context"
	"maps"
	"slices"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/gosource"
	"overgo/internal/loop"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
)

// cellPackages are the packages a cell's measured path runs, by its task: the capability mediacapability.Catalog binds
// the task to, or the inference session the text tasks share. A task not named here, or one whose package its recipe
// picks at run time (image and video generation, transcription, reconstruction, the training and projection proofs),
// reads the whole catalog (wholeCatalog): a cell measured when it need not be costs time, and one left out a decline
// unseen.
var cellPackages = map[recipe.Task]string{
	recipe.TaskInference: "overgo/internal/inference", recipe.TaskEmbedding: "overgo/internal/inference",
	recipe.TaskRerank: "overgo/internal/inference", recipe.TaskSpeculation: "overgo/internal/inference",
	recipe.TaskGeneration: "overgo/internal/thoughtbank", recipe.TaskForecast: "overgo/internal/seriesforecast",
	recipe.TaskTabular: "overgo/internal/tabularicl", recipe.TaskSeq2Seq: "overgo/internal/seq2seq",
	recipe.TaskSpeech: "overgo/internal/speechsynth", recipe.TaskActivityDetection: "overgo/internal/speechactivity",
	recipe.TaskImageEmbedding: "overgo/internal/vitencoder", recipe.TaskVQA: "overgo/internal/vqaserve",
}

// wholeCatalog is the package a cell reads whose task cellPackages does not name, and modulePrefix opens the
// repository's own import paths.
const (
	wholeCatalog = "overgo/internal/mediacapability"
	modulePrefix = "overgo/"
)

// commandDirectories are the commands cells run through, whose own files every cell reads.
var commandDirectories = []string{"cmd/validate", "cmd/recipe", "cmd/evaluate", "cmd/train", "cmd/longform"}

// reachedCells says which cells, of those registered, a change since the commit the scorecard record compared
// measured can move (loop.ReachedCells): each reads its task's package and that package's imports, and the commands
// cells run. Without a record every cell is reached.
func reachedCells(ctx context.Context, root string, compared *scorecard.Record, registered []ModelValidation) (func(ModelValidation) bool, error) {
	if compared == nil || compared.Commit == "" {
		return func(ModelValidation) bool { return true }, nil
	}
	changed, err := gitauthority.Query(ctx, root, "diff", "--name-only", compared.Commit, "HEAD")
	if err != nil {
		return nil, err
	}
	packages := append(slices.Collect(maps.Values(cellPackages)), wholeCatalog)
	selection, err := gosource.HostBuildClosure(root, packages...)
	if err != nil {
		return nil, err
	}
	cells := map[string][]string{}
	for _, cell := range registered {
		measured := cmp.Or(cellPackages[cell.Kind], wholeCatalog)
		directories := slices.Clone(commandDirectories)
		for _, imported := range append(selection.Dependencies[measured], measured) {
			if directory, own := strings.CutPrefix(imported, modulePrefix); own {
				directories = append(directories, directory)
			}
		}
		cells[cellKey(cell)] = directories
	}
	reached := loop.ReachedCells(cells, strings.Fields(string(changed)))
	return func(cell ModelValidation) bool { return slices.Contains(reached, cellKey(cell)) }, nil
}
