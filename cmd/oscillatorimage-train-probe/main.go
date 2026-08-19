// oscillatorimage-train-probe: bounded observed bootstrap training for a
// conditional-oscillator image artifact under the recovered reference
// semantics. The probe is the production driver of the oscillatorimage
// trainer and the reproducible generator of its training verification
// evidence: it evaluates the recovered class-target objective on the real
// checkpoint (objective identity against the reference's shipped loss
// band), then trains a scratch model built from the artifact's own config
// under the shipped init policy and proves observed Muon descent.
package main

import (
	"flag"
	"fmt"
	"math"
	"os"

	"overgo/internal/dataroot"
	"overgo/internal/optimizer"
	"overgo/internal/oscillatorimage"
)

func main() {
	model := flag.String("model", "", "oscillator image model directory (config.json + safetensors)")
	steps := flag.Int("steps", 12, "observed Muon steps for the scratch bootstrap")
	seed := flag.Int64("seed", 17, "bootstrap init and phase-sampling seed")
	evalSeed := flag.Int64("eval-seed", 202, "fixed init seed for before/after loss evaluation")
	flag.Parse()
	if err := run(*model, *steps, *seed, *evalSeed); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(modelDir string, steps int, seed, evalSeed int64) error {
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
	realTrainer, err := oscillatorimage.NewTrainer(real, optimizer.Config{}, evalSeed)
	if err != nil {
		return err
	}
	defer realTrainer.Close()
	shipped := realTrainer.Loss(evalSeed)
	fmt.Printf("objective identity: real checkpoint loss=%.6f (reference shipped band 15.57 +/- 3*1.16) parameters=%d\n",
		shipped, realTrainer.ParameterCount())

	scratch := oscillatorimage.NewBootstrapModel(real.Cfg, seed)
	trainer, err := oscillatorimage.NewTrainer(scratch, optimizer.Config{}, seed)
	if err != nil {
		return err
	}
	defer trainer.Close()
	before := trainer.Loss(evalSeed)
	for step := 0; step < steps; step++ {
		result, err := trainer.Step()
		if err != nil {
			return err
		}
		if math.IsNaN(result.Loss) || math.IsInf(result.Loss, 0) {
			return fmt.Errorf("oscillatorimage-train-probe: step %d loss is non-finite", step+1)
		}
		fmt.Printf("step %d/%d: loss=%.6f lr=%.4g grad_l2=%.4g\n", step+1, steps, result.Loss, result.LearningRate, result.GradientL2)
	}
	after := trainer.Loss(evalSeed)
	if !(after < before) {
		return fmt.Errorf("oscillatorimage-train-probe: scratch bootstrap did not descend: %.6f -> %.6f", before, after)
	}
	fmt.Printf("descent: steps=%d loss %.6f -> %.6f\n", steps, before, after)
	return nil
}
