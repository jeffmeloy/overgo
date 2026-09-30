package main

import (
	"context"
	"flag"
	"fmt"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/optimizer"
	"overgo/internal/oscillatorimage"
	"overgo/internal/overgodb"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// trainOscillatorImage is the oscillatorimage route: under a supervised
// session it measures the real checkpoint's loss on the evaluation seed,
// then trains a scratch bootstrap of the same configuration, refuses a run
// that does not descend, and publishes the trained bootstrap as a
// checkpoint.
func trainOscillatorImage(flags *flag.FlagSet, args []string) error {
	model := flags.String("model", "", "oscillator image model directory (config.json + safetensors)")
	steps := clioptions.IntOverride(flags, "steps", "required observed Muon steps")
	storePath := flags.String("store", "overgodb-store", "OvergoDB for the session observation, recipe authority and checkpoint record")
	stimulus := flags.String("stimulus", "", "committed declaration of the procedural dataset, grounding the recipe dataset")
	out := flags.String("out", "", "directory the trained bootstrap's checkpoint publishes to")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *model == "" || *stimulus == "" || *out == "" || *steps <= 0 {
		return fmt.Errorf("%s: model, stimulus, out, and positive steps required", flags.Name())
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	directory := roots.ResolveModelPath(*model)
	weights, err := trainingworkflow.IdentifyModelWeights(directory)
	if err != nil {
		return err
	}
	ctx := context.Background()
	store, err := overgodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	// The session's recipe is the real checkpoint's, whose configuration the
	// bootstrap takes; the shared run publishes no scratch initial state, so
	// the checkpoint names the real checkpoint as the model it trained from.
	session, err := openSession(ctx, store, weights, *stimulus)
	if err != nil {
		return err
	}
	real, err := oscillatorimage.Load(directory)
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
	return boundedRun{name: flags.Name(), plan: plan, session: session,
		step: func(int) (optimizer.ObservedStepResult, error) {
			result, err := trainer.Step()
			return optimizer.ObservedStepResult(result), err
		},
		loss: func() (float64, error) { return trainer.Loss(evalSeed), nil },
		// Every update draws one fresh record of the declared procedural
		// dataset.
		publish: func(ctx context.Context) error { return session.publishPlanned(ctx, *out, trainer, plan.Updates()) },
	}.train(ctx)
}
