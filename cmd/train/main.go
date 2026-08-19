// Command train executes native recipe-bound training.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/repodb"
	"overgo/internal/trainingworkflow"
)

func main() {
	clioptions.MainNamed("train", run)
}

func run() error {
	model := flag.String("model", "", "model directory (safetensors + config.json + tokenizer.json)")
	dataset := flag.String("dataset", "", "UTF-8 training dataset")
	output := flag.String("out", "", "output directory for the trained checkpoint")
	resume := flag.String("resume", "", "resume checkpoint directory")
	reference := flag.String("reference", "", "frozen reference model directory for DPO")
	scale := flag.Float64("objective-scale", 0, "RL objective scale")
	steps := flag.Int("steps", 1, "number of Muon update steps")
	maximumSequence := flag.Int("seq", 512, "maximum token sequence; nonpositive keeps all")
	learningRate := flag.Float64("lr", 0, "base learning rate; nonpositive derives n_params^-1/2")
	momentum := flag.Float64("momentum", 0.9, "Muon momentum")
	host := flag.Bool("host", false, "force host execution")
	freezeLexical := flag.Bool("freeze-lexical", false, "freeze tied embedding/head; requires CUDA resident training")
	maxWall := flag.Duration("max-wall", 30*time.Minute, "abort at a step boundary when the first measured step projects the run past this bound (0 disables)")
	storePath := flag.String("store", "repodb-store", "RepoDB containing the active training recipe and policies")
	recipeID := flag.String("recipe", "", "active training recipe artifact ID")
	flag.Parse()
	store, err := repodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	parsedRecipe, err := artifact.ParseID(*recipeID)
	if err != nil {
		return fmt.Errorf("training recipe: %w", err)
	}

	result, err := trainingworkflow.Execute(context.Background(), trainingworkflow.Request{
		Repository: store, Recipe: parsedRecipe,
		ModelDirectory: *model, DatasetPath: *dataset, OutputDirectory: *output,
		ResumeDirectory: *resume, ReferenceDirectory: *reference,
		Steps: *steps, MaximumSequence: *maximumSequence, LearningRate: *learningRate,
		Momentum: *momentum, ObjectiveScale: *scale, Host: *host, FreezeLexical: *freezeLexical,
		Progress: os.Stderr, MaxProjectedWall: *maxWall,
	})
	if err != nil {
		return err
	}
	count := len(result.Losses) + len(result.DPO) + len(result.GRPO)
	fmt.Printf("backend=%s objective=%s batches=%d stream_position=%d steps=%d lr=%s momentum=%g\n",
		result.Backend, result.Objective, count, result.StreamPosition, *steps, learningRateLabel(*learningRate), *momentum)
	for index, loss := range result.Losses {
		fmt.Printf("step %d: loss %.6f\n", index, loss)
	}
	for _, observation := range result.DPO {
		fmt.Printf("step %d: loss %.6f relative_margin %.6f\n", observation.Step, observation.Loss, observation.RelativeMargin)
	}
	for _, observation := range result.GRPO {
		fmt.Printf("step %d: loss %.6f mean_reward %.6f\n", observation.Step, observation.Loss, observation.MeanReward)
	}
	fmt.Printf("checkpoint=%s written to %s\n", result.Checkpoint.ID(), *output)
	return nil
}

func learningRateLabel(value float64) string {
	if value <= 0 {
		return "derived(n^-1/2)"
	}
	return fmt.Sprintf("%g", value)
}
