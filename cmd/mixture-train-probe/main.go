//overgo:runtime-inputs caller

// mixture-train-probe: bounded observed causal-LM training smoke for a
// densecausal artifact, including routed-mixture (DeepSeek-V2 family)
// checkpoints whose host graph is the only training path. The probe is the
// production driver of the mixture training graph and the reproducible
// generator of its verification evidence: it loads the artifact and its
// tokenizer, trains on the leading window of a text dataset through the
// shared Muon stepper with fully DERIVED hyperparameters (base LR
// n_params^-1/2, momentum from the CLT effective-samples rule), observes
// every step, arms a first-step projection guard, audits gradient coverage
// against the compiled plan, and proves descent by re-evaluating the same
// window's loss after the observed steps.
package main

import (
	"fmt"
	"math"
	"os"
	"slices"
	"time"

	"overgo/internal/checked"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/densecausal"
	"overgo/internal/hfbpe"
	"overgo/internal/optimizer"
	"overgo/internal/trainingprogram"
)

func main() {
	clioptions.TrainProbeMain(
		"model directory (safetensors + config.json + tokenizer.json)",
		"UTF-8 training dataset",
		"seq", "token window; omitted derives from model and dataset extents",
		run,
	)
}

func run(modelDir, datasetPath string, steps, seq int, maxWall time.Duration) error {
	if modelDir == "" || datasetPath == "" || steps <= 0 || seq < 0 {
		return fmt.Errorf("mixture-train-probe: model, dataset, positive steps, and nonnegative sequence override required")
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	directory := roots.ResolveModelPath(modelDir)
	loadStart := time.Now()
	model, err := densecausal.Load(directory)
	if err != nil {
		return err
	}
	tokenizer, err := hfbpe.Load(directory)
	if err != nil {
		return err
	}
	raw, err := os.ReadFile(datasetPath)
	if err != nil {
		return err
	}
	tokens, err := tokenizer.Encode(string(raw))
	if err != nil {
		return err
	}
	if seq == 0 {
		seq = len(tokens)
		for _, bound := range []int{model.Dims.ContextLength, model.Dims.AttentionWindow} {
			if bound > 0 {
				seq = min(seq, bound)
			}
		}
	}
	if seq < 2 {
		return fmt.Errorf("mixture-train-probe: derived token window %d cannot train next-token objective", seq)
	}
	if len(tokens) < seq {
		return fmt.Errorf("mixture-train-probe: dataset yields %d tokens, need %d", len(tokens), seq)
	}
	window := tokens[:seq]

	plan, err := model.TrainingPlan()
	if err != nil {
		return err
	}
	session, err := trainingprogram.CompileProbeSessionPlan(trainingprogram.ProbeSpec{
		Objective: trainingprogram.ObjectiveTokenPrediction, Updates: steps, MaximumSequence: seq,
		MaxProjectedWall: maxWall, Parameters: plan.ParameterCount(), Optimizer: trainingprogram.BuiltinOptimizerPolicy(),
	})
	if err != nil {
		return err
	}
	config := session.Optimizer()
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
	scatter := func() {
		for name, group := range groups {
			copy(model.Weights[name], weights[group.Start:group.End])
		}
	}
	stepper, err := optimizer.NewStepper(weights, gradients, plan, config)
	if err != nil {
		return err
	}
	defer stepper.Close()
	fmt.Printf("trainable parameters=%d groups=%d derived_lr=%.4g derived_momentum=%.4f mixture_top_k=%d window=%d load_wall=%s\n",
		plan.ParameterCount(), plan.GroupCount(), config.BaseLearningRate, config.Momentum, model.Dims.MoE.TopK, seq, time.Since(loadStart).Round(time.Millisecond))

	start := time.Now()
	var first float64
	for step := 0; step < session.Updates(); step++ {
		stepStart := time.Now()
		loss, _, grads, err := model.LossAndGrads(window)
		if err != nil {
			return err
		}
		if !checked.Finite64(loss) {
			return fmt.Errorf("mixture-train-probe: step %d loss is non-finite", step+1)
		}
		clear(gradients)
		var missing []string
		var gradientSquared float64
		for name, values := range grads {
			group, ok := groups[name]
			if !ok {
				missing = append(missing, name)
				continue
			}
			copy(gradients[group.Start:group.End], values)
			for _, g := range values {
				gradientSquared += float64(g) * float64(g)
			}
		}
		if len(missing) > 0 {
			slices.Sort(missing)
			return fmt.Errorf("mixture-train-probe: gradients outside the compiled plan: %v", missing)
		}
		if err := stepper.Step(); err != nil {
			return err
		}
		scatter()
		fmt.Printf("step %d/%d: loss=%.6f grad_l2=%.4g wall=%s\n",
			step+1, session.Updates(), loss, math.Sqrt(gradientSquared), time.Since(stepStart).Round(time.Millisecond))
		if step == 0 {
			first = loss
			if gradientSquared == 0 {
				return fmt.Errorf("mixture-train-probe: first step carried no gradient")
			}
			if err := session.AdmitStepWall(time.Since(start)); err != nil {
				return fmt.Errorf("mixture-train-probe: %w", err)
			}
		}
	}
	after, _, err := model.Loss(window)
	if err != nil {
		return err
	}
	if !checked.Finite64(after) || !(after < first) {
		return fmt.Errorf("mixture-train-probe: loss did not descend: before=%.6f after=%.6f", first, after)
	}
	fmt.Printf("descent: steps=%d span_tokens=%d loss %.6f -> %.6f total_wall=%s\n",
		session.Updates(), session.Updates()*seq, first, after, time.Since(start).Round(time.Millisecond))
	return nil
}
