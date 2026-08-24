// Command train executes native recipe-bound training.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/overgodb"
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
	scale := clioptions.Float64Override(flag.CommandLine, "objective-scale", "RL objective scale")
	steps := clioptions.IntOverride(flag.CommandLine, "steps", "Muon update steps; omitted derives from dataset units")
	maximumSequence := clioptions.IntOverride(flag.CommandLine, "seq", "maximum token sequence; omitted derives from model context")
	host := flag.Bool("host", false, "force host execution")
	freezeLexical := flag.Bool("freeze-lexical", false, "freeze tied embedding/head; requires CUDA resident training")
	maxWall := clioptions.DurationOverride(flag.CommandLine, "max-wall", "optional projected-wall bound")
	storePath := flag.String("store", "overgodb-store", "OvergoDB containing the active training recipe and policies")
	recipeID := flag.String("recipe", "", "active training recipe artifact ID")
	bootstrapRecipe := flag.String("bootstrap-recipe", "", "publish, verify, and activate a token-training recipe for the model weights file at this path, then exit")
	flag.Parse()
	store, err := overgodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	if *bootstrapRecipe != "" {
		return bootstrapTrainingRecipe(store, *bootstrapRecipe, *dataset)
	}
	parsedRecipe, err := artifact.ParseID(*recipeID)
	if err != nil {
		return fmt.Errorf("training recipe: %w", err)
	}

	result, err := trainingworkflow.Execute(context.Background(), trainingworkflow.Request{
		Repository: store, Recipe: parsedRecipe, Observations: store,
		ModelDirectory: *model, DatasetPath: *dataset, OutputDirectory: *output,
		ResumeDirectory: *resume, ReferenceDirectory: *reference,
		Steps: *steps, MaximumSequence: *maximumSequence,
		ObjectiveScale: *scale, Host: *host, FreezeLexical: *freezeLexical,
		Progress: os.Stderr, MaxProjectedWall: *maxWall,
	})
	if err != nil {
		return err
	}
	if result.Observation.Valid() {
		fmt.Printf("session observation: %s\n", result.Observation)
	}
	count := len(result.Losses) + len(result.DPO) + len(result.GRPO)
	fmt.Printf("backend=%s objective=%s batches=%d stream_position=%d steps=%d recipe_lr=%g recipe_momentum=%g\n",
		result.Backend, result.Objective, count, result.StreamPosition, result.Plan.Updates(),
		result.Optimizer.BaseLearningRate, result.Optimizer.Momentum)
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
