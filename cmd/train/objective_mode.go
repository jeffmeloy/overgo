package main

import (
	"context"
	"encoding/json"
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
	// SaveEvery, when positive, saves a checkpoint beside Out after every that many
	// updates but the last, Keep of the newest kept, and ResumeRun names one to
	// continue from, all on the host.
	SaveEvery, Keep int
	ResumeRun       string
	// Order, when not zero, seeds the order the run reads its training records in.
	Order uint64
	// Runs, when positive, trains that many runs from the same base, their order
	// seeds counting up from Order, and publishes them as a change corpus.
	Runs int
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
	// Quantized keeps the adapted projections of a GGUF base as the file stores them.
	Quantized bool
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
		// A continued adapter brings its own rank, alpha, targets and scaling; whether to keep its base quantized is the run's.
		return &trainingworkflow.LoRA{From: flags.From, Quantized: flags.Quantized}
	}
	return &trainingworkflow.LoRA{Rank: flags.Rank, Alpha: flags.Alpha, Targets: names, Seed: flags.Seed, RSLoRA: flags.RSLoRA, UpRate: flags.Plus, Quantized: flags.Quantized}
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
	if run.Runs > 0 {
		return runChangeCorpus(ctx, store, run, recipeID, output)
	}
	result, err := trainingworkflow.RunTextHeldout(ctx, store, run.request(recipeID, run.Order, run.Out, output))
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

// request is the held-out text run the flags ask for, reading its training
// records by order and publishing at out.
func (run objectiveRun) request(recipeID artifact.ID, order uint64, out string, output io.Writer) trainingworkflow.TextHeldoutRequest {
	return trainingworkflow.TextHeldoutRequest{
		Alias: run.Alias, Recipe: recipeID, ModelDirectory: run.Model, OutputDirectory: out, Steps: run.Steps, MaximumSequence: run.Sequence,
		Host: run.Host || run.LoRA != nil || run.Average > 0 || run.SaveEvery > 0 || run.ResumeRun != "", Device: denseDevice, FreezeLexical: run.FreezeLexical, MaxProjectedWall: run.MaxWall, Progress: output,
		Proof: run.Proof, LoRA: run.LoRA, Average: run.Average,
		CheckpointEvery: run.SaveEvery, CheckpointKeep: run.Keep, Resume: run.ResumeRun, Order: order,
	}
}

// firstOrderSeed is the order seed a corpus counts up from when the run gives
// none: zero is membership order, which is no seed.
const firstOrderSeed = 1

// runChangeCorpus trains run.Runs bounded runs of one base under one
// objective, their order seeds counting up from the run's, or from
// firstOrderSeed when it gives none, publishes them as a change corpus, and
// writes one line a run: the change and the verdict it earned.
func runChangeCorpus(ctx context.Context, store *overgodb.Store, run objectiveRun, recipeID artifact.ID, output io.Writer) error {
	corpus, err := trainingworkflow.CollectChangeCorpus(ctx, store, run.request(recipeID, 0, run.Out, output), max(run.Order, firstOrderSeed), run.Runs)
	if err != nil {
		return err
	}
	lines := json.NewEncoder(output)
	for _, change := range corpus.Changes {
		if err := lines.Encode(change); err != nil {
			return err
		}
	}
	_, err = fmt.Fprintf(output, "change corpus %s: %d changes of model %s under objective %s, judged by %s\n", corpus.ID, len(corpus.Changes), corpus.Model, corpus.Objective, corpus.Metric)
	return err
}

// proposalDirectory is where a proposal's candidates are written, under the
// directory its corpus was collected in.
const proposalDirectory = "proposal"

// proposeChange has a change proposed from the corpus whose runs sit under
// out and judged against its controls, and writes the judgment: a line for
// each candidate, then whether the proposal is kept and by what margin.
func proposeChange(ctx context.Context, store *overgodb.Store, corpus, model, out string, sequence int, output io.Writer) error {
	corpusID, err := artifact.ParseID(corpus)
	if err != nil {
		return fmt.Errorf("change corpus: %w", err)
	}
	judgment, err := trainingworkflow.ProposeChange(ctx, store, trainingworkflow.ChangeProposalRequest{
		Corpus: corpusID, ModelDirectory: model, RunsDirectory: out, OutputDirectory: filepath.Join(out, proposalDirectory), MaximumSequence: sequence, Progress: output,
	})
	if err != nil {
		return err
	}
	lines := json.NewEncoder(output)
	for _, candidate := range judgment.Candidates {
		if err := lines.Encode(candidate); err != nil {
			return err
		}
	}
	_, err = fmt.Fprintf(output, "change judgment %s: kept=%t; proposed less %s is %.6f, between %.6f and %.6f over the redrawn held-out records; a code of rank %d explains %.3f of a left-out run's gain; the runs' gains spread %.6f against a noise of %.6f\n",
		judgment.ID, judgment.Kept, judgment.Control, judgment.Difference, judgment.Lower, judgment.Upper, judgment.Rank, judgment.Explained, judgment.GainSpread, judgment.GainNoise)
	return err
}

// reportVerdict writes what a held-out run published.
func reportVerdict(output io.Writer, result trainingworkflow.HeldoutResult) error {
	_, err := fmt.Fprintf(output, "held-out verdict %s passed=%t observation=%s trained=%s view=%s plan=%s baseline=%s candidate=%s\n",
		result.Verdict, result.Passed, result.Observation, result.Trained, result.View, result.Plan, result.Baseline, result.Candidate)
	return err
}
