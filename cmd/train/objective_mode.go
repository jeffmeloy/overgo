package main

import (
	"context"
	"fmt"
	"io"
	"strings"
	"time"

	"overgo/internal/overgodb"
	"overgo/internal/trainingworkflow"
)

// objectiveRun is cmd/train's -objective mode: train a text model on the
// registered objective's training split, publish the trained model at Out,
// and publish the held-out verdict that binds it. Table and forecast
// objectives train through their routes.
type objectiveRun struct {
	Alias, Model, Out string
	Steps, Sequence   int
	Host              bool
	MaxWall           time.Duration
}

// objectiveConflictingFlags are the flags an -objective run cannot honour,
// sorted: the objective's registered dataset and split replace the raw
// dataset, and its recipe every recipe-session input.
var objectiveConflictingFlags = []string{
	"audio-manifest", "bootstrap-recipe", "dataset", "freeze-lexical", "objective-scale",
	"preference-margin", "preference-mode", "preview-dataset", "recipe", "reference", "resume",
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
	if run.Model == "" || run.Out == "" || run.Steps <= 0 {
		return fmt.Errorf("train: -objective requires -model, -out and positive -steps")
	}
	result, err := trainingworkflow.RunTextHeldout(ctx, store, trainingworkflow.TextHeldoutRequest{
		Alias: run.Alias, ModelDirectory: run.Model, OutputDirectory: run.Out, Steps: run.Steps, MaximumSequence: run.Sequence,
		Host: run.Host, Device: denseDevice, MaxProjectedWall: run.MaxWall,
	})
	if err != nil {
		return err
	}
	return reportVerdict(output, result)
}

// reportVerdict writes what a held-out run published.
func reportVerdict(output io.Writer, result trainingworkflow.HeldoutResult) error {
	_, err := fmt.Fprintf(output, "held-out verdict %s passed=%t observation=%s trained=%s view=%s plan=%s baseline=%s candidate=%s\n",
		result.Verdict, result.Passed, result.Observation, result.Trained, result.View, result.Plan, result.Baseline, result.Candidate)
	return err
}
