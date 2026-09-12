//go:build windows

// hybrid-stack-train-probe: bounded observed full-stack training smoke for a
// real hybrid artifact (mixed full-attention + gated-delta decoder stack).
// The probe is the production driver of the host-master streamed training
// lane and the reproducible generator of its verification evidence: it binds
// EVERY layer of the artifact as f32 masters decoded from the serving
// storage, derives the stack geometry and the Muon hyperparameters entirely
// from the artifact's own declarations (base LR from the parameter count,
// momentum from the CLT effective-samples rule, constant schedule — no free
// knobs), trains the Model's hidden-state objective on real token rows, and
// requires a strictly decreasing finite loss trajectory. A first-step wall
// projection guard aborts runs that cannot finish inside -max-wall.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/clioptions"
	"overgo/internal/cuda/device"
	"overgo/internal/gguf"
	"overgo/internal/hybridtrain"
	"overgo/internal/model"
	"overgo/internal/overgodb"
	"overgo/internal/processmeasure"
	"overgo/internal/quant"
	"overgo/internal/runrecord"
	"overgo/internal/tensor"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

type servingFixture struct {
	Cases []struct {
		Name      string   `json:"name"`
		PromptIDs []uint32 `json:"prompt_ids"`
	} `json:"cases"`
}

// promptTokens returns the named case's token IDs from the serving fixture.
func promptTokens(path, caseName string) ([]uint32, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var fixture servingFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	for _, item := range fixture.Cases {
		if item.Name == caseName {
			if len(item.PromptIDs) < 2 {
				return nil, fmt.Errorf("fixture case %q has %d token IDs, need at least 2", caseName, len(item.PromptIDs))
			}
			return item.PromptIDs, nil
		}
	}
	return nil, fmt.Errorf("fixture %s has no case %q", path, caseName)
}

func run() error {
	modelPath := flag.String("model", "", "GGUF artifact path")
	fixturePath := flag.String("fixture", filepath.Join("fixtures", "qwen35_4b_serving_golden.json"), "serving fixture holding the prompt token IDs")
	caseName := flag.String("case", "capital", "fixture case whose prompt IDs feed the objective")
	steps := clioptions.IntOverride(flag.CommandLine, "steps", "required observed training steps")
	maxWall := clioptions.DurationOverride(flag.CommandLine, "max-wall", "optional projected-wall bound")
	inspect := flag.Bool("inspect", false, "print the artifact census and exit without training")
	storePath := flag.String("store", "overgodb-store", "OvergoDB for the session observation and recipe authority")
	lane := flag.String("lane", "full-slab", "training lane: full-slab (masters+gradients+momentum all full host slabs) or layer-streamed (bounded host triplication: gradients never exceed one layer)")
	roundTrip := flag.Bool("serving-roundtrip", false, "after training, requantize layer 0's mlp gate masters to their declared serving dtype and re-evaluate the objective through the round trip")
	flag.Parse()
	if *lane != "full-slab" && *lane != "layer-streamed" {
		return fmt.Errorf("-lane must be full-slab or layer-streamed, got %q", *lane)
	}
	if *modelPath == "" {
		return fmt.Errorf("-model is required")
	}
	if *inspect {
		return printCensus(*modelPath)
	}
	if *steps < tensor.PairedExtent {
		return fmt.Errorf("-steps must be at least %d to observe a loss decrease", tensor.PairedExtent)
	}

	tokens, err := promptTokens(*fixturePath, *caseName)
	if err != nil {
		return err
	}
	fmt.Printf("artifact: %s\n", *modelPath)
	fmt.Printf("objective tokens: case=%q ids=%v (%d prediction rows)\n", *caseName, tokens, len(tokens)-1)

	// The whole run executes as a model-session: recipe authority, then lease
	// admission BEFORE any weight touches memory.
	ctx := context.Background()
	store, err := overgodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	modelFile, err := os.Open(*modelPath)
	if err != nil {
		return err
	}
	modelID, _, err := artifact.Identify(artifact.KindModel, modelFile)
	modelFile.Close()
	if err != nil {
		return err
	}
	recipeID, err := trainingworkflow.BootstrapTokenRecipe(ctx, store, *modelPath, *fixturePath)
	if err != nil {
		return err
	}
	observer, err := trainingworkflow.NewObserver(store, false)
	if err != nil {
		return err
	}
	if err := observer.Admit(ctx, modelID, recipeID); err != nil {
		return err
	}
	authority, err := trainingworkflow.ResolveSessionAuthority(ctx, store, modelID, recipeID)
	if err != nil {
		return err
	}
	fmt.Printf("session admitted: model=%s recipe=%s\n", modelID, recipeID)

	loadStart := time.Now()
	trained, err := hybridtrain.LoadStackArtifact(context.Background(), *modelPath, tokens)
	if err != nil {
		return err
	}
	loadWall := time.Since(loadStart)
	attention, recurrent := 0, 0
	for _, kind := range trained.Cfg.Types {
		if kind == hybridtrain.LinearAttention {
			recurrent++
		} else {
			attention++
		}
	}
	total := trained.MatrixParamCount() + trained.VectorParamCount()
	session, err := trainingprogram.CompileProbeSessionPlan(trainingprogram.ProbeSpec{
		Objective: authority.Objective, Updates: *steps, MaximumSequence: len(tokens),
		MaxProjectedWall: *maxWall, Parameters: total, Optimizer: authority.Optimizer,
	})
	if err != nil {
		return err
	}
	config := session.Optimizer()
	fmt.Printf("stack census: %d layers (%d attention, %d recurrent)\n", len(trained.Cfg.Types), attention, recurrent)
	fmt.Printf("trainable parameters: %d matrix + %d vector = %d total\n",
		trained.MatrixParamCount(), trained.VectorParamCount(), total)
	fmt.Printf("derived hyperparameters: baseLR=%.6g momentum=%.6g schedule=constant (no free knobs)\n",
		config.BaseLearningRate, config.Momentum)
	fmt.Printf("load wall: %s  program=%s\n", loadWall.Round(time.Millisecond), trained.Program().ID())
	observer.Phase(runrecord.PhaseLoad, loadWall)
	runtime.GC()
	if *lane == "layer-streamed" {
		budget, err := trained.LayerStreamedBudget()
		if err != nil {
			return err
		}
		const f32 = 4
		fmt.Printf("memory budget (%s, bounded host triplication):\n", *lane)
		fmt.Printf("  masters       %14d bytes (%d f32, full host slab — the forward needs it)\n", f32*budget.MasterElems, budget.MasterElems)
		fmt.Printf("  momentum      %14d bytes (%d f32, full host slab — persists across steps)\n", f32*budget.MomentumElems, budget.MomentumElems)
		fmt.Printf("  grad scratch  %14d bytes (largest layer, %d f32 — gradients NEVER triplicate)\n", f32*budget.ScratchElems, budget.ScratchElems)
		fmt.Printf("  vectors       %14d bytes (%d f32, params+grads+momentum host-side)\n", 3*f32*budget.VectorElems, budget.VectorElems)
		fmt.Printf("  bounded total %14d bytes vs %d bytes full triplication\n",
			f32*(2*budget.MasterElems+budget.ScratchElems+3*budget.VectorElems),
			f32*(3*budget.MasterElems+3*budget.VectorElems))
	}
	printPeakRSS("after artifact load")

	worker, err := device.New(device.DefaultOrdinal())
	if err != nil {
		return err
	}
	defer worker.Close()

	train := trained.TrainHostMasterStreamed
	if *lane == "layer-streamed" {
		train = trained.TrainHostMasterLayerStreamed
	}
	trainStart := time.Now()
	previous := trainStart
	trajectory, err := train(worker, session.Updates(), config, hybridtrain.TrainingOptions{Observe: func(step int, loss float64) error {
		now := time.Now()
		wall := now.Sub(previous)
		previous = now
		observer.SampleStep()
		fmt.Printf("step %d/%d  loss=%.6f  wall=%s\n", step+1, session.Updates(), loss, wall.Round(time.Millisecond))
		if !checked.Finite64(loss) {
			return fmt.Errorf("step %d loss is not finite: %g", step+1, loss)
		}
		if step == tensor.FirstOffset {
			projected := time.Duration(session.Updates()) * wall
			fmt.Printf("projection: %d steps x %s = %s (guard %s)\n",
				session.Updates(), wall.Round(time.Millisecond), projected.Round(time.Second), session.MaxProjectedWall())
			if err := session.AdmitStepWall(wall); err != nil {
				return err
			}
		}
		return nil
	}})
	trainWall := time.Since(trainStart)
	observer.Phase(runrecord.PhaseForwardBackward, trainWall)
	runErr := err
	if runErr == nil {
		for index, loss := range trajectory {
			if !checked.Finite64(loss) {
				runErr = fmt.Errorf("trajectory[%d] is not finite: %g", index, loss)
				break
			}
			if index > tensor.FirstOffset && loss >= trajectory[index-tensor.SingletonExtent] {
				runErr = fmt.Errorf("loss did not decrease at step %d: %v", index+tensor.SingletonExtent, trajectory)
				break
			}
		}
	}
	if runErr == nil && *roundTrip {
		runErr = servingRoundTrip(*modelPath, trained)
	}
	printPeakRSS("after training")
	streamed := uint64(session.Updates()) * uint64(len(tokens))
	observation, observeErr := observer.Finish(ctx, modelID, recipeID, runErr, streamed)
	if runErr != nil {
		return runErr
	}
	if observeErr != nil {
		return observeErr
	}
	fmt.Printf("trajectory: %v\n", trajectory)
	fmt.Printf("session observation: %s\n", observation)
	fmt.Printf("PASS: finite loss decreased over %d full-stack steps in %s\n", len(trajectory), trainWall.Round(time.Millisecond))
	return nil
}

// printPeakRSS reports the process RSS high-water mark; measurement failure is
// reported, never fatal — the run's own evidence stays primary.
func printPeakRSS(stage string) {
	peak, err := processmeasure.SelfPeakWorkingSet()
	if err != nil {
		fmt.Printf("process peak working set (%s): unavailable: %v\n", stage, err)
		return
	}
	fmt.Printf("process peak working set (%s): %d bytes\n", stage, peak)
}

// servingRoundTrip mirrors the organ cell's serving bar at full-stack scope:
// layer 0's trained mlp-gate masters requantize back to their DECLARED serving
// dtype (read from the artifact, never assumed) and the run's own objective is
// re-evaluated through the round trip, so the reported movement belongs to the
// serving-storage tensor rather than a float shadow.
func servingRoundTrip(modelPath string, trained *hybridtrain.Model) error {
	file, err := gguf.Open(modelPath)
	if err != nil {
		return err
	}
	defer file.Close()
	spec, err := model.ReadSpec(file)
	if err != nil {
		return err
	}
	weights, err := model.ReadWeights(file, spec)
	if err != nil {
		return err
	}
	if len(weights.Layers) == 0 || weights.Layers[0].FeedForwardGate == nil || len(trained.Weights) == 0 {
		return fmt.Errorf("serving round trip: layer 0 mlp gate is absent")
	}
	info := weights.Layers[0].FeedForwardGate
	gate := trained.Weights[0].MLP.Gate
	lossTrained := trained.EvaluateLoss()
	packed, err := quant.Quantize(info.Type, gate)
	if err != nil {
		return fmt.Errorf("serving round trip: requantize %q to %s: %w", info.Name, info.Type, err)
	}
	restored, err := quant.Dequantize(info.Type, packed, uint64(len(gate)))
	if err != nil {
		return fmt.Errorf("serving round trip: decode %q: %w", info.Name, err)
	}
	copy(gate, restored)
	lossRoundTrip := trained.EvaluateLoss()
	fmt.Printf("serving round trip (%s -> %s -> f32): trained-masters loss %.6f -> requantized %.6f (delta %+.6g)\n",
		info.Name, info.Type, lossTrained, lossRoundTrip, lossRoundTrip-lossTrained)
	if !checked.Finite64(lossRoundTrip) {
		return fmt.Errorf("serving round trip: loss is not finite: %g", lossRoundTrip)
	}
	return nil
}

func main() { clioptions.MainNamed("hybrid-stack-train-probe", run) }
