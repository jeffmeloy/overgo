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
	"math"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/cuda/device"
	"overgo/internal/dataroot"
	"overgo/internal/hybridtrain"
	"overgo/internal/optimizer"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
	"overgo/internal/trainingsession"
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
	defaultModel := ""
	if roots, err := dataroot.ResolveCurrent(); err == nil {
		defaultModel = filepath.Join(roots.Checkpoints, "overgo-hfconvert", "Qwen3.5-4B-f16.gguf")
	}
	modelPath := flag.String("model", defaultModel, "GGUF artifact path (defaults to the dataroot-resolved Qwen3.5-4B checkpoint)")
	fixturePath := flag.String("fixture", filepath.Join("fixtures", "qwen35_4b_serving_golden.json"), "serving fixture holding the prompt token IDs")
	caseName := flag.String("case", "capital", "fixture case whose prompt IDs feed the objective")
	steps := flag.Int("steps", 3, "observed training steps")
	maxWall := flag.Duration("max-wall", 25*time.Minute, "abort when the first step projects the run past this wall")
	inspect := flag.Bool("inspect", false, "print the artifact census and exit without training")
	storePath := flag.String("store", "repodb-store", "RepoDB for the session observation and recipe authority")
	flag.Parse()
	if *modelPath == "" {
		return fmt.Errorf("-model is required (dataroot did not resolve)")
	}
	if *inspect {
		return printCensus(*modelPath)
	}
	if *steps < 2 {
		return fmt.Errorf("-steps must be at least 2 to observe a loss decrease")
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
	store, err := repodb.Open(*storePath)
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
	recipeID, err := trainingsession.BootstrapTokenRecipe(ctx, store, *modelPath, *fixturePath)
	if err != nil {
		return err
	}
	observer, err := trainingsession.New(store, false)
	if err != nil {
		return err
	}
	if err := observer.Admit(ctx, modelID, recipeID); err != nil {
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
	config := optimizer.Config{
		BaseLearningRate: optimizer.DeriveBaseLR(total),
		Momentum:         optimizer.DeriveMomentum(),
		Schedule:         optimizer.ScheduleConstant,
	}
	fmt.Printf("stack census: %d layers (%d attention, %d recurrent)\n", len(trained.Cfg.Types), attention, recurrent)
	fmt.Printf("trainable parameters: %d matrix + %d vector = %d total\n",
		trained.MatrixParamCount(), trained.VectorParamCount(), total)
	fmt.Printf("derived hyperparameters: baseLR=%.6g momentum=%.6g schedule=constant (no free knobs)\n",
		config.BaseLearningRate, config.Momentum)
	fmt.Printf("load wall: %s  program=%s\n", loadWall.Round(time.Millisecond), trained.Program().ID())
	observer.Phase(runrecord.PhaseLoad, loadWall)
	runtime.GC()

	worker, err := device.New(0)
	if err != nil {
		return err
	}
	defer worker.Close()

	trainStart := time.Now()
	previous := trainStart
	trajectory, err := trained.TrainHostMasterStreamed(worker, *steps, config, func(step int, loss float64) error {
		now := time.Now()
		wall := now.Sub(previous)
		previous = now
		observer.SampleStep()
		fmt.Printf("step %d/%d  loss=%.6f  wall=%s\n", step+1, *steps, loss, wall.Round(time.Millisecond))
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			return fmt.Errorf("step %d loss is not finite: %g", step+1, loss)
		}
		if step == 0 {
			projected := time.Duration(*steps) * wall
			fmt.Printf("projection: %d steps x %s = %s (guard %s)\n",
				*steps, wall.Round(time.Millisecond), projected.Round(time.Second), *maxWall)
			if projected > *maxWall {
				return fmt.Errorf("projected wall %s exceeds -max-wall %s; aborting after the first step",
					projected.Round(time.Second), *maxWall)
			}
		}
		return nil
	})
	trainWall := time.Since(trainStart)
	observer.Phase(runrecord.PhaseForwardBackward, trainWall)
	runErr := err
	if runErr == nil {
		for index, loss := range trajectory {
			if math.IsNaN(loss) || math.IsInf(loss, 0) {
				runErr = fmt.Errorf("trajectory[%d] is not finite: %g", index, loss)
				break
			}
			if index > 0 && loss >= trajectory[index-1] {
				runErr = fmt.Errorf("loss did not decrease at step %d: %v", index+1, trajectory)
				break
			}
		}
	}
	streamed := uint64(*steps) * uint64(len(tokens))
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

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "hybrid-stack-train-probe:", err)
		os.Exit(1)
	}
}
