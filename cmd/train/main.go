// Command train executes native recipe-bound training.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/overgodb"
	"overgo/internal/strictjson"
	"overgo/internal/trainingworkflow"
)

func main() {
	clioptions.MainNamed("train", run)
}

func run() error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
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
	previewDataset := flag.String("preview-dataset", "", "validate one speech training example from a source/split/policy JSON manifest without loading a model or updating parameters")
	audioManifest := flag.String("audio-manifest", "", "CPU CTC adapter source-processing manifest; the active training recipe owns the registered dataset and split")
	flag.Parse()
	store, err := overgodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	if *previewDataset != "" {
		if *bootstrapRecipe != "" || *recipeID != "" || *model != "" || *output != "" || *resume != "" || *audioManifest != "" {
			return fmt.Errorf("train: -preview-dataset cannot be combined with model execution or recipe bootstrap")
		}
		return previewTrainingData(ctx, store, *previewDataset, os.Stdout)
	}
	if *bootstrapRecipe != "" {
		if *audioManifest != "" {
			return fmt.Errorf("train: -audio-manifest cannot be combined with recipe bootstrap")
		}
		return bootstrapTrainingRecipe(store, *bootstrapRecipe, *dataset)
	}
	parsedRecipe, err := artifact.ParseID(*recipeID)
	if err != nil {
		return fmt.Errorf("training recipe: %w", err)
	}
	var audio *trainingworkflow.AudioTrainingSpec
	if *audioManifest != "" {
		data, err := artifact.ReadContentFile(*audioManifest)
		if err != nil {
			return err
		}
		audio = new(trainingworkflow.AudioTrainingSpec)
		if err := strictjson.DecodeBytes(data, audio); err != nil {
			return err
		}
	}

	result, err := trainingworkflow.Execute(ctx, trainingworkflow.Request{
		Audio:      audio,
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
	if result.Candidate.Valid() {
		fmt.Printf("candidate=%s (not activated)\n", result.Candidate)
	}
	return nil
}
