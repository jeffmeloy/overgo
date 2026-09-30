package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/hostoptimizer"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// TestEveryTrainRoutePublishesACheckpoint holds every route to publishing
// what it trains: each takes a -out checkpoint target it refuses to train
// without, and the shared run it publishes through trains under the shared
// guards and writes a checkpoint that loads with its weights and is
// recorded in the store.
func TestEveryTrainRoutePublishesACheckpoint(t *testing.T) {
	for name := range trainRoutes {
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
// bootstrapped session, publishes it as the routes publish a planned
// trainer, and reads the checkpoint back from the directory and the store.
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
	plan, err := probeSession(session.authority.Objective, 3, testMatrix.End, 0)
	if err != nil {
		t.Fatal(err)
	}
	matrix := newHostMatrix(t, session.authority.Objective, plan.Optimizer())
	target := filepath.Join(root, "checkpoint")
	run := boundedRun{name: "shared-run", plan: plan, session: session, loss: matrix.loss,
		step:    func(int) (optimizer.ObservedStepResult, error) { return matrix.step(), nil },
		publish: func(ctx context.Context) error { return session.publishPlanned(ctx, target, matrix, plan.Updates()) },
	}
	if err := run.train(ctx); err != nil {
		t.Fatal(err)
	}
	stream, err := trainingworkflow.TextStream([]byte(dataset), uint64(plan.Updates()))
	if err != nil {
		t.Fatal(err)
	}
	checkpoint, err := trainingprogram.LoadCheckpoint(target)
	if err != nil {
		t.Fatal(err)
	}
	_, recorded, err := store.Artifact(ctx, checkpoint.ID())
	if err != nil || !recorded || checkpoint.Model != session.model || checkpoint.Program != matrix.program.ID() ||
		checkpoint.Stream != trainingprogram.DatasetState(stream) {
		t.Fatalf("checkpoint %s of %s under program %s at %+v is recorded %t (%v), want %s's checkpoint under %s at %+v recorded",
			checkpoint.ID(), checkpoint.Model, checkpoint.Program, checkpoint.Stream, recorded, err, session.model, matrix.program.ID(), stream)
	}
	rerun := filepath.Join(root, "rerun")
	if err := session.publishPlanned(ctx, rerun, matrix, plan.Updates()); err != nil {
		t.Fatal(err)
	}
	if again, err := trainingprogram.LoadCheckpoint(rerun); err != nil || again.ID() != checkpoint.ID() {
		t.Fatalf("a rerun publishing the same checkpoint elsewhere = (%s, %v), want %s recorded again", again.ID(), err, checkpoint.ID())
	}
}

// testMatrix is the one 2x2 group a hostMatrix trains.
var testMatrix = hostoptimizer.GroupSpec{Name: "adapter.weight", End: 4, Rows: 2, Cols: 2}

// hostMatrix is the smallest checkpointed and planned trainer: a host Muon
// matrix stepped toward all ones under a compiled objective program.
type hostMatrix struct {
	plan               hostoptimizer.Plan
	program            trainingprogram.TrainingProgram
	muon               *hostoptimizer.Optimizer
	weights, gradients []float32
}

func newHostMatrix(t *testing.T, objective trainingprogram.ObjectiveKind, config hostoptimizer.Config) *hostMatrix {
	t.Helper()
	plan, err := hostoptimizer.CompilePlan(testMatrix.End, []hostoptimizer.GroupSpec{testMatrix})
	if err != nil {
		t.Fatal(err)
	}
	matrix := &hostMatrix{plan: plan, weights: make([]float32, testMatrix.End), gradients: make([]float32, testMatrix.End)}
	if matrix.program, err = trainingprogram.CompileObjectiveProgram(objective, nil, plan); err != nil {
		t.Fatal(err)
	}
	if matrix.muon, err = hostoptimizer.New(matrix.weights, matrix.gradients, plan, config); err != nil {
		t.Fatal(err)
	}
	return matrix
}

func (m *hostMatrix) loss() (float64, error) {
	var total float64
	for _, weight := range m.weights {
		total += (float64(weight) - 1) * (float64(weight) - 1)
	}
	return total / float64(len(m.weights)), nil
}

func (m *hostMatrix) step() optimizer.ObservedStepResult {
	before, _ := m.loss()
	for index, weight := range m.weights {
		m.gradients[index] = 2 * (weight - 1) / float32(len(m.weights))
	}
	result := m.muon.Step()
	return optimizer.ObservedStepResult{Loss: before, Step: result.Step, LearningRate: result.LearningRate, GradientL2: result.GradientL2}
}

func (m *hostMatrix) Program() trainingprogram.TrainingProgram { return m.program }

func (m *hostMatrix) OptimizerPlan() optimizer.Plan { return m.plan }

func (m *hostMatrix) Snapshot() optimizer.State { return m.muon.Snapshot() }

func (m *hostMatrix) SaveWeights(path string) error {
	return optimizer.SavePlanWeights(path, m.plan, m.weights)
}

// publishUnderObjective steps a host matrix under an objective kind's
// program and publishes it through the recipe a held-out run's session
// activated, as the forecast and table routes do, then holds the checkpoint
// to loading with that program and being recorded in the store.
func publishUnderObjective(t *testing.T, ctx context.Context, store *overgodb.Store, recipeID artifact.ID, kind trainingprogram.ObjectiveKind, stream artifact.ID) error {
	t.Helper()
	config, err := trainingprogram.BuiltinOptimizerPolicy().Config(testMatrix.End)
	if err != nil {
		return err
	}
	matrix := newHostMatrix(t, kind, config)
	result := matrix.step()
	target := filepath.Join(t.TempDir(), "checkpoint")
	if err := publishTrained(ctx, store, recipeID, target, matrix, trainingdata.StreamState{Identity: stream, Position: uint64(result.Step)}); err != nil {
		return err
	}
	loaded, err := trainingprogram.LoadCheckpoint(target)
	if err != nil {
		return err
	}
	if _, recorded, err := store.Artifact(ctx, loaded.ID()); err != nil || !recorded || loaded.Program != matrix.program.ID() {
		t.Fatalf("checkpoint %s of program %s recorded %t (%v), want %s's checkpoint recorded", loaded.ID(), loaded.Program, recorded, err, matrix.program.ID())
	}
	return nil
}
