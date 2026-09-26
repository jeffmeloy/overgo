package main

import (
	"context"
	"fmt"
	"io"
	"strings"
	"time"

	"overgo/internal/overgodb"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// objectiveRun is cmd/train's -objective mode: train the model the
// objective's kind names (a text model, or a TabFM table head) on the
// registered objective's training split and publish its held-out verdict.
type objectiveRun struct {
	Alias, Model    string
	Steps, Sequence int
	Host            bool
	MaxWall         time.Duration
}

// objectiveConflictingFlags are the flags an -objective run cannot honour,
// sorted: the objective's registered dataset and split replace the raw
// dataset, its recipe and every recipe-session input, and the run publishes
// a verdict, not a checkpoint.
var objectiveConflictingFlags = []string{
	"audio-manifest", "bootstrap-recipe", "dataset", "freeze-lexical", "objective-scale",
	"out", "preview-dataset", "recipe", "reference", "resume",
}

// objectiveModeConflicts names, sorted, the flags in set that an -objective
// run cannot honour.
func objectiveModeConflicts(set map[string]bool) []string {
	var conflicts []string
	for _, name := range objectiveConflictingFlags {
		if set[name] {
			conflicts = append(conflicts, "-"+name)
		}
	}
	return conflicts
}

func runObjectiveMode(ctx context.Context, store *overgodb.Store, run objectiveRun, set map[string]bool, output io.Writer) error {
	if conflicts := objectiveModeConflicts(set); len(conflicts) != 0 {
		return fmt.Errorf("train: -objective trains on the objective's registered split and publishes a verdict; it cannot be combined with %s", strings.Join(conflicts, ", "))
	}
	if run.Model == "" || run.Steps <= 0 {
		return fmt.Errorf("train: -objective requires -model and positive -steps")
	}
	objective, err := trainingworkflow.LoadObjectiveAlias(ctx, store, run.Alias)
	if err != nil {
		return err
	}
	var result trainingworkflow.HeldoutResult
	if objective.Kind == trainingprogram.ObjectiveTablePrediction {
		result, err = trainTableObjective(ctx, store, run, objective)
	} else {
		result, err = trainingworkflow.RunTextHeldout(ctx, store, trainingworkflow.TextHeldoutRequest{
			Alias: run.Alias, ModelDirectory: run.Model, Steps: run.Steps, MaximumSequence: run.Sequence,
			Host: run.Host, MaxProjectedWall: run.MaxWall,
		})
	}
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "held-out verdict %s passed=%t observation=%s view=%s plan=%s baseline=%s candidate=%s\n",
		result.Verdict, result.Passed, result.Observation, result.View, result.Plan, result.Baseline, result.Candidate)
	return err
}
