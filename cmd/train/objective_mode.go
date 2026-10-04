package main

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/trainingworkflow"
)

// objectiveRun is cmd/train's -objective mode: train a text model on the
// registered objective's training split, publish the trained model at Out,
// and publish the held-out verdict that binds it. A proof run re-verifies
// the recipe by training alone and publishes no verdict. Table and forecast
// objectives train through their routes.
type objectiveRun struct {
	Alias, Model, Out          string
	Recipe                     string
	Steps, Sequence            int
	Host, FreezeLexical, Proof bool
	MaxWall                    time.Duration
	// LoRA, when set, trains an adapter over the frozen model instead of the
	// model, on the host.
	LoRA *trainingworkflow.LoRA
	// Average, when positive, is the decay of a moving average of the weights
	// the run publishes as the trained model, on the host.
	Average float64
}

// loraFlags are the -lora flags as parsed.
type loraFlags struct {
	Rank    int
	Alpha   float64
	Targets string
	Seed    uint64
	From    string
	RSLoRA  bool
	Plus    float64
}

// loraRequest is the LoRA run the -lora flags ask for: none when they ask for
// none, an adapter of the given rank otherwise, or the one in the PEFT
// directory from. Alpha defaults to trainingworkflow.LoRAAlphaPerRank times the rank; targets
// are comma-separated projection names, all linear projections when empty.
func loraRequest(flags loraFlags) *trainingworkflow.LoRA {
	if flags.Rank <= 0 && flags.From == "" {
		return nil
	}
	if flags.Alpha == 0 {
		flags.Alpha = float64(flags.Rank * trainingworkflow.LoRAAlphaPerRank)
	}
	var names []string
	for name := range strings.SplitSeq(flags.Targets, ",") {
		if name = strings.TrimSpace(name); name != "" {
			names = append(names, name)
		}
	}
	if flags.From != "" {
		// A continued adapter brings its own rank, alpha, targets and scaling.
		return &trainingworkflow.LoRA{From: flags.From}
	}
	return &trainingworkflow.LoRA{Rank: flags.Rank, Alpha: flags.Alpha, Targets: names, Seed: flags.Seed, RSLoRA: flags.RSLoRA, UpRate: flags.Plus}
}

// objectiveConflictingFlags are the flags an -objective run cannot honour,
// sorted: the objective's registered dataset and split replace the raw
// dataset and raw session inputs. A selected recipe must bind that objective.
var objectiveConflictingFlags = []string{
	"audio-manifest", "bootstrap-recipe", "dataset", "objective-scale",
	"preference-margin", "preference-mode", "preview-dataset", "reference", "resume",
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
	var recipeID artifact.ID
	if run.Recipe != "" {
		var err error
		if recipeID, err = artifact.ParseID(run.Recipe); err != nil {
			return fmt.Errorf("training recipe: %w", err)
		}
	}
	result, err := trainingworkflow.RunTextHeldout(ctx, store, trainingworkflow.TextHeldoutRequest{
		Alias: run.Alias, Recipe: recipeID, ModelDirectory: run.Model, OutputDirectory: run.Out, Steps: run.Steps, MaximumSequence: run.Sequence,
		Host: run.Host || run.LoRA != nil || run.Average > 0, Device: denseDevice, FreezeLexical: run.FreezeLexical, MaxProjectedWall: run.MaxWall, Progress: output,
		Proof: run.Proof, LoRA: run.LoRA, Average: run.Average,
	})
	if err != nil {
		return err
	}
	if run.Proof {
		_, err = fmt.Fprintf(output, "training proof observation=%s trained=%s; no held-out verdict published\n", result.Observation, result.Trained)
		return err
	}
	if run.LoRA != nil {
		if _, err := fmt.Fprintf(output, "LoRA adapter written to %s as a PEFT directory\n", filepath.Join(run.Out, trainingworkflow.LoRAAdapterDirectory)); err != nil {
			return err
		}
	}
	return reportVerdict(output, result)
}

// reportVerdict writes what a held-out run published.
func reportVerdict(output io.Writer, result trainingworkflow.HeldoutResult) error {
	_, err := fmt.Fprintf(output, "held-out verdict %s passed=%t observation=%s trained=%s view=%s plan=%s baseline=%s candidate=%s\n",
		result.Verdict, result.Passed, result.Observation, result.Trained, result.View, result.Plan, result.Baseline, result.Candidate)
	return err
}
