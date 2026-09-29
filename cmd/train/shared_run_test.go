package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/hostoptimizer"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/safetensors"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// checkpointless names each route whose trainer has no weight serializer
// yet, by the trainer holding its trained weights. Every other route takes a
// -out checkpoint target and publishes what it trains there through the
// shared run.
var checkpointless = map[string]string{
	"dit":             "latentvideo.DiTTrainer (internal/latentvideo/dittrain.go:49)",
	"flow-organ":      "latentimage.FinalLayerTrainer (internal/latentimage/train.go:113), routedlm.FlowHeadTrainer (internal/routedlm/flowtrain.go:37), routedlm.LatentBridgeTrainer (internal/routedlm/latentbridge.go:66) and the ternary masters (cmd/train/route_flow_organ.go)",
	"graft":           "composition.RunViability's per-seed bridges (internal/composition/viability.go:65)",
	"mixture":         "optimizer.Stepper, whose momentum no checkpoint can read (internal/optimizer/stepper.go:4)",
	"mot":             "routedlm.ModalityTransformerTrainer (internal/routedlm/mottrain.go:52)",
	"oscillatorimage": "oscillatorimage.Trainer (internal/oscillatorimage/trainer.go:61)",
	"seriesforecast":  "seriesforecast.Trainer (internal/seriesforecast/train.go:38) and its optimizer.Stepper, whose momentum no checkpoint can read (internal/optimizer/stepper.go:4)",
	"tabular":         "tabularicl.Trainer (internal/tabularicl/train.go:29) and its optimizer.Stepper, whose momentum no checkpoint can read (internal/optimizer/stepper.go:4)",
}

// TestEveryTrainRoutePublishesACheckpoint holds every route to publishing
// what it trains: a route either names its trainer as not yet serializable
// or takes a -out checkpoint target it refuses to train without, and the
// shared run it publishes through trains under the shared guards and writes
// a checkpoint that loads with its weights and is recorded in the store.
func TestEveryTrainRoutePublishesACheckpoint(t *testing.T) {
	for name, trainer := range checkpointless {
		if _, routed := trainRoutes[name]; !routed || trainer == "" {
			t.Errorf("checkpointless names %q with trainer %q: each entry is a route and names the trainer holding its weights", name, trainer)
		}
	}
	for name := range trainRoutes {
		if _, unserializable := checkpointless[name]; unserializable {
			continue
		}
		t.Run(name, func(t *testing.T) {
			if err := runRoute([]string{name, "-out", t.TempDir()}); err == nil || strings.Contains(err.Error(), "not defined") {
				t.Errorf("route %s with only a checkpoint target = %v, want its refusal of the missing flags", name, err)
			}
			if err := runRoute([]string{name}); err == nil || !strings.Contains(err.Error(), "out") {
				t.Errorf("route %s without a checkpoint target = %v, want a refusal requiring -out", name, err)
			}
		})
	}
	t.Run("shared-run", sharedRunPublishes)
}

// sharedRunPublishes trains a host Muon matrix toward a fixed target under a
// bootstrapped session, publishes it, and reads the checkpoint back from the
// directory and the store.
func sharedRunPublishes(t *testing.T) {
	root, dataset := t.TempDir(), "training data"
	modelPath, datasetPath := filepath.Join(root, "model.gguf"), filepath.Join(root, "dataset.txt")
	for path, data := range map[string]string{modelPath: "weights", datasetPath: dataset} {
		if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	session, err := openSession(ctx, store, modelPath, datasetPath)
	if err != nil {
		t.Fatal(err)
	}
	matrix := hostoptimizer.GroupSpec{Name: "adapter.weight", End: 4, Rows: 2, Cols: 2}
	muonPlan, err := hostoptimizer.CompilePlan(matrix.End, []hostoptimizer.GroupSpec{matrix})
	if err != nil {
		t.Fatal(err)
	}
	program, err := trainingprogram.CompileObjectiveProgram(session.authority.Objective, nil, muonPlan)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := probeSession(session.authority.Objective, 3, matrix.End, 0)
	if err != nil {
		t.Fatal(err)
	}
	weights, gradients := make([]float32, matrix.End), make([]float32, matrix.End)
	muon, err := hostoptimizer.New(weights, gradients, muonPlan, plan.Optimizer())
	if err != nil {
		t.Fatal(err)
	}
	loss := func() (float64, error) {
		var total float64
		for _, weight := range weights {
			total += (float64(weight) - 1) * (float64(weight) - 1)
		}
		return total / float64(len(weights)), nil
	}
	run := boundedRun{name: "shared-run", plan: plan, session: session, loss: loss,
		step: func(int) (optimizer.ObservedStepResult, error) {
			before, _ := loss()
			for index, weight := range weights {
				gradients[index] = 2 * (weight - 1) / float32(len(weights))
			}
			result := muon.Step()
			return optimizer.ObservedStepResult{Loss: before, Step: result.Step, LearningRate: result.LearningRate, GradientL2: result.GradientL2}, nil
		},
	}
	if err := run.descend(); err != nil {
		t.Fatal(err)
	}
	stream, err := trainingworkflow.TextStream([]byte(dataset), uint64(plan.Updates()))
	if err != nil {
		t.Fatal(err)
	}
	write := func(stage string) error {
		return safetensors.Save(filepath.Join(stage, trainingprogram.CheckpointWeights),
			map[string][]float32{matrix.Name: weights}, map[string][]int{matrix.Name: {matrix.Rows, matrix.Cols}}, nil)
	}
	target := filepath.Join(root, "checkpoint")
	checkpoint, err := session.publish(ctx, target, program, muon.Snapshot(), stream, write)
	if err := session.finish(ctx, plan, err); err != nil {
		t.Fatal(err)
	}
	loaded, loadErr := trainingprogram.LoadCheckpoint(target)
	_, recorded, err := store.Artifact(ctx, checkpoint.ID())
	if loadErr != nil || err != nil || loaded.ID() != checkpoint.ID() || loaded.Model != session.model || !recorded {
		t.Fatalf("checkpoint %s loads as (%s for %s, %v) and is recorded %t (%v), want it loaded for %s and recorded",
			checkpoint.ID(), loaded.ID(), loaded.Model, loadErr, recorded, err, session.model)
	}
	if again, err := session.publish(ctx, filepath.Join(root, "rerun"), program, muon.Snapshot(), stream, write); err != nil || again.ID() != checkpoint.ID() {
		t.Fatalf("a rerun publishing the same checkpoint elsewhere = (%s, %v), want %s recorded again", again.ID(), err, checkpoint.ID())
	}
}
