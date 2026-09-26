package main

import (
	"flag"
	"fmt"

	"overgo/internal/checked"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
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
	plan, err := trainingprogram.CompileProbeSessionPlan(trainingprogram.ProbeSpec{
		Objective: trainingprogram.ObjectiveLatentL2, Updates: *steps,
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
			return fmt.Errorf("%s: step %d loss is non-finite", flags.Name(), step+1)
		}
		fmt.Printf("step %d/%d: loss=%.6f lr=%.4g grad_l2=%.4g\n", step+1, plan.Updates(), result.Loss, result.LearningRate, result.GradientL2)
	}
	after := trainer.Loss(evalSeed)
	if !(after < before) {
		return fmt.Errorf("%s: scratch bootstrap did not descend: %.6f -> %.6f", flags.Name(), before, after)
	}
	fmt.Printf("descent: steps=%d loss %.6f -> %.6f\n", plan.Updates(), before, after)
	return nil
}
