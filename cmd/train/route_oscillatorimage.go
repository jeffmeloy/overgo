package main

import (
	"context"
	"flag"
	"fmt"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/optimizer"
	"overgo/internal/oscillatorimage"
	"overgo/internal/trainingprogram"
)

// trainOscillatorImage is the oscillatorimage route: it measures the real
// checkpoint's loss on the evaluation seed, then trains a scratch bootstrap
// of the same configuration and refuses a run that does not descend.
func trainOscillatorImage(flags *flag.FlagSet, args []string) error {
	model := flags.String("model", "", "oscillator image model directory (config.json + safetensors)")
	steps := clioptions.IntOverride(flags, "steps", "required observed Muon steps")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *model == "" || *steps <= 0 {
		return fmt.Errorf("%s: -model is required and -steps must be positive", flags.Name())
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	real, err := oscillatorimage.Load(roots.ResolveModelPath(*model))
	if err != nil {
		return err
	}
	plan, err := probeSession(trainingprogram.ObjectiveLatentL2, *steps, real.TrainableParameterCount(), 0)
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
	return boundedRun{name: flags.Name(), plan: plan,
		step: func(int) (optimizer.ObservedStepResult, error) {
			result, err := trainer.Step()
			return optimizer.ObservedStepResult(result), err
		},
		loss: func() (float64, error) { return trainer.Loss(evalSeed), nil },
	}.train(context.Background())
}
