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

// The adapter route steps through the CUDA worker, which only the Windows
// build carries, so it registers itself here.
func init() { trainRoutes["adapter"] = trainAdapter }

// adapterTraining is what the adapter route reports: the layer trained, the
// example rows, the adapter's parameters, every step's loss and where the
// weights were written.
type adapterTraining struct {
	Layer      uint32    `json:"layer"`
	Rows       int       `json:"rows"`
	Parameters int       `json:"parameters"`
	Losses     []float64 `json:"losses"`
	Weights    string    `json:"weights,omitzero"`
}

// trainAdapter is the adapter route: bounded artifact-declared adapter
// training of one layer of a GGUF model on teacher-forced text.
func trainAdapter(flags *flag.FlagSet, args []string) (err error) {
	modelPath := flags.String("model", "", "GGUF model")
	text := flags.String("text", "", "teacher-forced text")
	layer := flags.Uint("layer", 0, "adapter layer")
	steps := clioptions.IntOverride(flags, "steps", "required update steps")
	weights := flags.String("out", "", "optional adapter safetensors")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *modelPath == "" || *text == "" || *steps <= 0 || uint64(*layer) > uint64(^uint32(0)) {
		return fmt.Errorf("%s: model, text, valid layer, and positive steps required", flags.Name())
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
	result := adapterTraining{Layer: uint32(*layer), Rows: example.Rows, Parameters: trained.ParameterCount(), Losses: make([]float64, plan.Updates()), Weights: *weights}
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
