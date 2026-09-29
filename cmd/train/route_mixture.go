package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/densecausal"
	"overgo/internal/export"
	"overgo/internal/hfbpe"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/safetensors"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// shortestNextTokenWindow is the fewest tokens a next-token objective trains
// on: one input and the target after it.
const shortestNextTokenWindow = 2

// trainMixture is the mixture route: bounded observed causal-LM training of
// a densecausal artifact, routed-mixture (DeepSeek-V2 family) checkpoints
// included, whose host graph is their only training path. It is the
// production driver of the mixture training graph and the reproducible
// generator of its verification evidence: it loads the artifact and its
// tokenizer, trains on the leading window of a text dataset through the
// shared Muon stepper with derived hyperparameters (base LR n_params^-1/2,
// momentum from the CLT effective-samples rule), observes every step, arms a
// first-step projection guard, audits gradient coverage against the compiled
// plan, proves descent by re-evaluating the same window's loss after the
// observed steps, and publishes the trained model as a checkpoint, all under
// a supervised session.
func trainMixture(flags *flag.FlagSet, args []string) error {
	modelDir := flags.String("model", "", "model directory (safetensors + config.json + tokenizer.json)")
	datasetPath := flags.String("dataset", "", "UTF-8 training dataset")
	steps := clioptions.IntOverride(flags, "steps", "required observed Muon steps")
	seq := clioptions.IntOverride(flags, "seq", "token window; omitted derives from model and dataset extents")
	maxWall := clioptions.DurationOverride(flags, "max-wall", "optional projected-wall bound")
	out := flags.String("out", "", "directory the trained model's checkpoint publishes to")
	storePath := flags.String("store", "overgodb-store", "OvergoDB for the session observation, recipe authority and checkpoint record")
	if err := flags.Parse(args); err != nil {
		return err
	}
	name := flags.Name()
	if *modelDir == "" || *datasetPath == "" || *out == "" || *steps <= 0 || *seq < 0 {
		return fmt.Errorf("%s: model, dataset, out, positive steps, and nonnegative sequence override required", name)
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	directory := roots.ResolveModelPath(*modelDir)
	weightsPath, err := trainingworkflow.IdentifyModelWeights(directory)
	if err != nil {
		return err
	}
	ctx := context.Background()
	store, err := overgodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	session, err := openSession(ctx, store, weightsPath, *datasetPath)
	if err != nil {
		return err
	}
	loadStart := time.Now()
	model, err := densecausal.Load(directory)
	if err != nil {
		return err
	}
	tokenizer, err := hfbpe.Load(directory)
	if err != nil {
		return err
	}
	raw, err := os.ReadFile(*datasetPath)
	if err != nil {
		return err
	}
	tokens, err := tokenizer.Encode(string(raw))
	if err != nil {
		return err
	}
	window := *seq
	if window == 0 {
		window = len(tokens)
		for _, bound := range []int{model.Dims.ContextLength, model.Dims.AttentionWindow} {
			if bound > 0 {
				window = min(window, bound)
			}
		}
	}
	if window < shortestNextTokenWindow {
		return fmt.Errorf("%s: derived token window %d cannot train next-token objective", name, window)
	}
	if len(tokens) < window {
		return fmt.Errorf("%s: dataset yields %d tokens, need %d", name, len(tokens), window)
	}
	trainingWindow := tokens[:window]

	plan, err := model.TrainingPlan()
	if err != nil {
		return err
	}
	program, err := trainingprogram.CompileObjectiveProgram(session.authority.Objective, nil, plan)
	if err != nil {
		return err
	}
	runPlan, err := trainingprogram.CompileProbeSessionPlan(trainingprogram.ProbeSpec{
		Objective: session.authority.Objective, Updates: *steps, MaximumSequence: window,
		MaxProjectedWall: *maxWall, Parameters: plan.ParameterCount(), Optimizer: session.authority.Optimizer,
	})
	if err != nil {
		return err
	}
	config := runPlan.Optimizer()
	weights := make([]float32, plan.ParameterCount())
	gradients := make([]float32, plan.ParameterCount())
	// The flat buffers are optimizer storage only: the model's compiled layer
	// bindings alias its own weight slices, so updates scatter back into
	// those slices after every step rather than repointing the map.
	groups := make(map[string]optimizer.GroupSpec, plan.GroupCount())
	for index := 0; index < plan.GroupCount(); index++ {
		group, _ := plan.Group(index)
		groups[group.Name] = group.GroupSpec
		copy(weights[group.Start:group.End], model.Weights[group.Name])
	}
	stepper, err := optimizer.NewStepper(weights, gradients, plan, config)
	if err != nil {
		return err
	}
	defer stepper.Close()
	fmt.Printf("trainable parameters=%d groups=%d derived_lr=%.4g derived_momentum=%.4f mixture_top_k=%d window=%d load_wall=%s\n",
		plan.ParameterCount(), plan.GroupCount(), config.BaseLearningRate, config.Momentum, model.Dims.MoE.TopK, window, time.Since(loadStart).Round(time.Millisecond))

	var updates int
	return boundedRun{name: name, plan: runPlan, session: session,
		step: func(int) (optimizer.ObservedStepResult, error) {
			loss, _, grads, err := model.LossAndGrads(trainingWindow)
			if err != nil {
				return optimizer.ObservedStepResult{}, err
			}
			clear(gradients)
			var missing []string
			var gradientSquared float64
			for group, values := range grads {
				span, ok := groups[group]
				if !ok {
					missing = append(missing, group)
					continue
				}
				copy(gradients[span.Start:span.End], values)
				for _, g := range values {
					gradientSquared += float64(g) * float64(g)
				}
			}
			if len(missing) > 0 {
				slices.Sort(missing)
				return optimizer.ObservedStepResult{}, fmt.Errorf("%s: gradients outside the compiled plan: %v", name, missing)
			}
			result, err := optimizer.Advance(stepper, config, &updates, loss, math.Sqrt(gradientSquared))
			for tensor, span := range groups {
				copy(model.Weights[tensor], weights[span.Start:span.End])
			}
			return result, err
		},
		loss: func() (float64, error) {
			loss, _, err := model.Loss(trainingWindow)
			return loss, err
		},
		// The dataset is the one record every update trains on, so the run
		// consumed as many records as it took updates; every model tensor
		// is a plan group, so the checkpoint holds the whole trained model,
		// with its config and tokenizer, as the dense workflow's does.
		publish: func(ctx context.Context) error {
			stream, err := trainingworkflow.TextStream(raw, uint64(updates))
			if err != nil {
				return err
			}
			checkpoint, err := session.publish(ctx, *out, program, stepper.Snapshot(), stream, func(stage string) error {
				if err := safetensors.Save(filepath.Join(stage, trainingprogram.CheckpointWeights), model.Weights, model.Shapes, nil); err != nil {
					return err
				}
				return export.CopyFiles(directory, stage, []string{"config.json", "tokenizer.json"})
			})
			if err == nil {
				fmt.Printf("checkpoint=%s written to %s\n", checkpoint.ID(), *out)
			}
			return err
		},
	}.train(ctx)
}
