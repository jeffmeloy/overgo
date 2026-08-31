// Command adapter-train-probe runs bounded artifact-declared adapter training.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"

	"overgo/internal/adaptertrain"
	"overgo/internal/clioptions"
	"overgo/internal/cuda/device"
	"overgo/internal/gguf"
	"overgo/internal/tokenizer"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

type output struct {
	Layer      uint32    `json:"layer"`
	Rows       int       `json:"rows"`
	Parameters int       `json:"parameters"`
	Losses     []float64 `json:"losses"`
	Weights    string    `json:"weights,omitzero"`
}

func main() { clioptions.MainNamed("adapter-train-probe", run) }

func run() (err error) {
	modelPath := flag.String("model", "", "GGUF model")
	text := flag.String("text", "", "teacher-forced text")
	layer := flag.Uint("layer", 0, "adapter layer")
	steps := clioptions.IntOverride(flag.CommandLine, "steps", "required update steps")
	weights := flag.String("out", "", "optional adapter safetensors")
	flag.Parse()
	if *modelPath == "" || *text == "" || *steps <= 0 || uint64(*layer) > uint64(^uint32(0)) {
		return fmt.Errorf("model, text, valid layer, and positive steps required")
	}
	file, err := gguf.Open(*modelPath)
	if err != nil {
		return err
	}
	vocabulary, loadErr := tokenizer.Load(file)
	err = errors.Join(loadErr, file.Close())
	if err != nil {
		return err
	}
	ids, err := vocabulary.Encode(*text, tokenizer.EncodeOptions{AddSpecial: true})
	if err != nil {
		return err
	}
	input, target, err := trainingdata.AdjacentTokenRows(ids)
	if err != nil {
		return err
	}
	ctx := context.Background()
	policy := trainingprogram.BuiltinOptimizerPolicy()
	trained, _, err := adaptertrain.LoadArtifact(ctx, *modelPath, uint32(*layer), policy)
	if err != nil {
		return err
	}
	plan, err := trainingprogram.CompileProbeSessionPlan(trainingprogram.ProbeSpec{
		Objective: trainingprogram.ObjectiveTokenPrediction, Updates: *steps, MaximumSequence: len(ids),
		Parameters: trained.ParameterCount(), Optimizer: policy,
	})
	if err != nil {
		return err
	}
	example, err := trained.BuildExample(ctx, *modelPath, nil, input, target, "text")
	if err != nil {
		return err
	}
	worker, err := device.New(device.DefaultOrdinal())
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, worker.Close()) }()
	result := output{Layer: uint32(*layer), Rows: example.Rows, Parameters: trained.ParameterCount(), Losses: make([]float64, plan.Updates()), Weights: *weights}
	for index := range result.Losses {
		result.Losses[index], err = trained.Step(worker, example)
		if err != nil {
			return err
		}
	}
	if *weights != "" {
		if err := trained.SaveWeights(*weights); err != nil {
			return err
		}
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(result)
}
