// Command train executes native recipe-bound training.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"overgo/internal/clioptions"
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
	scale := flag.Float64("dpo-scale", 0, "DPO scale; required with -reference")
	steps := flag.Int("steps", 1, "number of Muon update steps")
	maximumSequence := flag.Int("seq", 512, "maximum token sequence; nonpositive keeps all")
	learningRate := flag.Float64("lr", 0, "base learning rate; nonpositive derives n_params^-1/2")
	momentum := flag.Float64("momentum", 0.9, "Muon momentum")
	host := flag.Bool("host", false, "force host execution")
	freezeLexical := flag.Bool("freeze-lexical", false, "freeze tied embedding/head; requires CUDA resident training")
	maxWall := flag.Duration("max-wall", 30*time.Minute, "abort at a step boundary when the first measured step projects the run past this bound (0 disables)")
	flag.Parse()

	result, err := trainingworkflow.Execute(context.Background(), trainingworkflow.Request{
		ModelDirectory: *model, DatasetPath: *dataset, OutputDirectory: *output,
		ResumeDirectory: *resume, ReferenceDirectory: *reference,
		Steps: *steps, MaximumSequence: *maximumSequence, LearningRate: *learningRate,
		Momentum: *momentum, DPOScale: *scale, Host: *host, FreezeLexical: *freezeLexical,
		Progress: os.Stderr, MaxProjectedWall: *maxWall,
	})
	if err != nil {
		return err
	}
	count := len(result.Losses) + len(result.DPO)
	fmt.Printf("backend=%s objective=%s batches=%d stream_position=%d steps=%d lr=%s momentum=%g\n",
		result.Backend, result.Objective, count, result.StreamPosition, *steps, learningRateLabel(*learningRate), *momentum)
	for index, loss := range result.Losses {
		fmt.Printf("step %d: loss %.6f\n", index, loss)
	}
	for _, observation := range result.DPO {
		fmt.Printf("step %d: loss %.6f relative_margin %.6f\n", observation.Step, observation.Loss, observation.RelativeMargin)
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
