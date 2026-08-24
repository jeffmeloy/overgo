// oscillatorimage-train-probe: bounded artifact and bootstrap training evidence.
package main

import (
	"flag"
	"fmt"
	"os"

	"overgo/internal/checked"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/oscillatorimage"
	"overgo/internal/trainingprogram"
)

func main() {
	model := flag.String("model", "", "oscillator image model directory (config.json + safetensors)")
	steps := clioptions.IntOverride(flag.CommandLine, "steps", "required observed Muon steps")
	flag.Parse()
	if err := run(*model, *steps); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(modelDir string, steps int) error {
	if modelDir == "" || steps <= 0 {
		return fmt.Errorf("oscillatorimage-train-probe: -model is required and -steps must be positive")
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	real, err := oscillatorimage.Load(roots.ResolveModelPath(modelDir))
	if err != nil {
		return err
	}
	plan, err := trainingprogram.CompileProbeSessionPlan(trainingprogram.ProbeSpec{
		Objective: trainingprogram.ObjectiveLatentL2, Updates: steps,
		Parameters: real.TrainableParameterCount(), Optimizer: trainingprogram.BuiltinOptimizerPolicy(),
	})
	if err != nil {
		return err
	}
	config := plan.Optimizer()
	seed, err := plan.Seed("bootstrap")
	if err != nil {
		return err
	}
	evalSeed, err := plan.Seed("evaluation")
	if err != nil {
		return err
	}
	realTrainer, err := oscillatorimage.NewTrainer(real, config, evalSeed)
	if err != nil {
		return err
	}
	defer realTrainer.Close()
	shipped := realTrainer.Loss(evalSeed)
	fmt.Printf("objective identity: real checkpoint loss=%.6f (reference shipped band 15.57 +/- 3*1.16) parameters=%d\n",
		shipped, realTrainer.ParameterCount())

	scratch := oscillatorimage.NewBootstrapModel(real.Cfg, seed)
	trainer, err := oscillatorimage.NewTrainer(scratch, config, seed)
	if err != nil {
		return err
	}
	defer trainer.Close()
	before := trainer.Loss(evalSeed)
	for step := 0; step < plan.Updates(); step++ {
		result, err := trainer.Step()
		if err != nil {
			return err
		}
		if !checked.Finite64(result.Loss) {
			return fmt.Errorf("oscillatorimage-train-probe: step %d loss is non-finite", step+1)
		}
		fmt.Printf("step %d/%d: loss=%.6f lr=%.4g grad_l2=%.4g\n", step+1, plan.Updates(), result.Loss, result.LearningRate, result.GradientL2)
	}
	after := trainer.Loss(evalSeed)
	if !(after < before) {
		return fmt.Errorf("oscillatorimage-train-probe: scratch bootstrap did not descend: %.6f -> %.6f", before, after)
	}
	fmt.Printf("descent: steps=%d loss %.6f -> %.6f\n", plan.Updates(), before, after)
	return nil
}
