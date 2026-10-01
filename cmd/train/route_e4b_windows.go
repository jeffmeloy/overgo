package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/cuda/device"
	"overgo/internal/e4btrain"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/trainingprogram"
)

// The e4b route steps through the CUDA worker, which only the Windows build
// carries, so it registers itself here.
func init() { trainRoutes["e4b"] = trainE4B }

// trainE4B is the e4b route: bounded full-parameter training of a Gemma-4
// E4B artifact on the leading token window of a text dataset, as a
// supervised session whose trained decoder publishes as a checkpoint. Every
// decoder, adapter and norm tensor trains as an f32 master under the Muon
// hyperparameters the session's optimizer policy derives from the
// parameter count; the windowed attention VJP and the Muon step run on the
// device.
func trainE4B(flags *flag.FlagSet, args []string) (err error) {
	modelPath := flags.String("model", "", "GGUF artifact path")
	datasetPath := flags.String("dataset", "", "UTF-8 teacher-forced text dataset")
	steps := clioptions.IntOverride(flags, "steps", "required observed Muon steps")
	seq := clioptions.IntOverride(flags, "seq", "token window; omitted trains on every token of the dataset")
	maxWall := clioptions.DurationOverride(flags, "max-wall", "optional projected-wall bound")
	out := flags.String("out", "", "directory the trained decoder's checkpoint publishes to")
	storePath := flags.String("store", "overgodb-store", "OvergoDB for the session observation, recipe authority and checkpoint record")
	if err := flags.Parse(args); err != nil {
		return err
	}
	name := flags.Name()
	if *modelPath == "" || *datasetPath == "" || *out == "" || *steps <= 0 || *seq < 0 {
		return fmt.Errorf("%s: model, dataset, out, positive steps, and nonnegative sequence override required", name)
	}
	window, err := datasetWindow(*modelPath, *datasetPath, *seq)
	if err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	ctx := context.Background()
	store, err := overgodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	session, err := openSession(ctx, store, *modelPath, *datasetPath)
	if err != nil {
		return err
	}
	loadStarted := time.Now()
	model, err := e4btrain.LoadArtifact(ctx, *modelPath, window)
	if err != nil {
		return err
	}
	plan, err := trainingprogram.CompileProbeSessionPlan(trainingprogram.ProbeSpec{
		Objective: session.authority.Objective, Updates: *steps, MaximumSequence: len(window),
		MaxProjectedWall: *maxWall, Parameters: model.ParameterCount(), Optimizer: session.authority.Optimizer,
	})
	if err != nil {
		return err
	}
	session.observer.Phase(runrecord.PhaseLoad, time.Since(loadStarted))
	worker, err := device.New(device.DefaultOrdinal())
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, worker.Close()) }()
	config := plan.Optimizer()
	trainer, err := e4btrain.NewTrainer(worker, model, config)
	if err != nil {
		return err
	}
	fmt.Printf("full-parameter e4b: parameters=%d tensors=%d window=%d derived_lr=%.6g derived_momentum=%.6g load_wall=%s\n",
		model.ParameterCount(), model.OptimizerPlan().GroupCount(), len(window), config.BaseLearningRate, config.Momentum,
		time.Since(loadStarted).Round(time.Millisecond))
	return boundedRun{name: name, plan: plan, session: session,
		step: func(int) (optimizer.ObservedStepResult, error) { return trainer.Step() },
		loss: trainer.Loss,
		// The window is the one record every update trains on.
		publish: func(ctx context.Context) error {
			return session.publishPlanned(ctx, *out, trainer, plan.Updates())
		},
	}.train(ctx)
}

// datasetWindow is the dataset's leading token window under the artifact's
// own tokenizer: every token, or the first seq when seq is set.
func datasetWindow(modelPath, datasetPath string, seq int) ([]uint32, error) {
	_, ids, err := datasetTokens(modelPath, datasetPath)
	if err != nil {
		return nil, err
	}
	if seq > 0 {
		ids = ids[:min(len(ids), seq)]
	}
	if len(ids) < shortestNextTokenWindow {
		return nil, fmt.Errorf("the dataset yields %d tokens, a window needs %d", len(ids), shortestNextTokenWindow)
	}
	window := make([]uint32, len(ids))
	for index, id := range ids {
		if id < 0 {
			return nil, fmt.Errorf("token %d of the dataset has a negative identity", index)
		}
		window[index] = uint32(id)
	}
	return window, nil
}
