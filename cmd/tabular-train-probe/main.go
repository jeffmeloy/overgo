// Command tabular-train-probe runs bounded decoder training.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/tabularicl"
	"overgo/internal/trainingprogram"
)

type output struct {
	Task       string                       `json:"task"`
	Parameters int                          `json:"parameters"`
	Steps      []tabularicl.TrainStepResult `json:"steps"`
}

func main() { clioptions.MainNamed("tabular-train-probe", run) }

func run() (err error) {
	model := flag.String("model", "", "TabFM model directory")
	task := flag.String("task", "", "classification or regression")
	input := flag.String("input", "", "training request JSON")
	steps := clioptions.IntOverride(flag.CommandLine, "steps", "required decoder update steps")
	flag.Parse()
	if *model == "" || *task == "" || *input == "" || *steps <= 0 {
		return fmt.Errorf("model, task, input, and positive steps required")
	}
	var request tabularicl.Request
	if err := jsonfile.Decode(*input, &request); err != nil {
		return err
	}
	if request.Task != *task {
		return fmt.Errorf("request task %q differs from %q", request.Task, *task)
	}
	head, err := tabularicl.LoadHead(filepath.Join(*model, *task))
	if err != nil {
		return err
	}
	plan, err := trainingprogram.CompileProbeSessionPlan(trainingprogram.ProbeSpec{
		Objective: trainingprogram.ObjectiveTablePrediction, Updates: *steps,
		Parameters: head.TrainableParameterCount(), Optimizer: trainingprogram.BuiltinOptimizerPolicy(),
	})
	if err != nil {
		return err
	}
	trainer, err := tabularicl.NewTrainer(head, plan.Optimizer())
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, trainer.Close()) }()
	result := output{Task: *task, Parameters: trainer.ParameterCount(), Steps: make([]tabularicl.TrainStepResult, plan.Updates())}
	for index := range result.Steps {
		result.Steps[index], err = trainer.Step(request)
		if err != nil {
			return err
		}
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(result)
}
