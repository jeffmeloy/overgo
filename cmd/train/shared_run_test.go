package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/hostoptimizer"
	"overgo/internal/optimizer"
	"overgo/internal/oscillatorimage"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
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
			if err := runRoute(t.Context(), []string{name, "-out", t.TempDir()}); err == nil || strings.Contains(err.Error(), "not defined") {
				t.Errorf("route %s with only a checkpoint target = %v, want its refusal of the missing flags", name, err)
			}
			if err := runRoute(t.Context(), []string{name}); err == nil || !strings.Contains(err.Error(), "out") {
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
	root, store, modelPath, datasetPath := routeFixture(t)
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
		step: func(int) (optimizer.ObservedStepResult, error) { return matrix.step(), nil },
		publish: func(ctx context.Context) error {
			return session.publishPlanned(ctx, target, matrix, plan.Updates())
		},
	}
	if err := run.train(ctx); err != nil {
		t.Fatal(err)
	}
	stream, err := trainingworkflow.TextStream([]byte(routeFixtureDataset), uint64(plan.Updates()))
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

// routeFixtureDataset is the dataset a route fixture's session grounds.
const routeFixtureDataset = "training data"

// routeFixture writes a stand-in model file and a dataset a route's session
// grounds, and opens a store for the session.
func routeFixture(t *testing.T) (root string, store *overgodb.Store, modelPath, datasetPath string) {
	t.Helper()
	root = t.TempDir()
	modelPath, datasetPath = filepath.Join(root, "model.safetensors"), filepath.Join(root, "dataset.txt")
	for path, data := range map[string]string{modelPath: "weights", datasetPath: routeFixtureDataset} {
		if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	return root, store, modelPath, datasetPath
}

// TestRoutePublicationRecordsActualAuthority holds the shared route
// publisher to the facts of what ran. A trainer that steps flow matching to
// video opens its session under that contract, so its checkpoint names the
// flow-matching program it executed and the dataset as the session read it,
// though the file changed after. A bootstrap built from nothing, as the
// oscillatorimage route trains, steps latent L2 toward images from its
// scratch construction, and its checkpoint names that construction as the
// model it trained from. A trainer whose objective the session's recipe
// does not declare is refused before any step and publishes nothing. A run
// whose context ends stops between updates, publishes nothing and is
// recorded as cancelled, and a setup that fails after admission finishes
// its session once.
func TestRoutePublicationRecordsActualAuthority(t *testing.T) {
	flow := generationContract(trainingprogram.ObjectiveFlowMatching, recipecontract.ModalityText, recipecontract.ModalityVideo, trainingprogram.MetricVideoPSNRUnit)
	t.Run("scratch start", func(t *testing.T) {
		root, store, _, datasetPath := routeFixture(t)
		declaration, err := os.ReadFile(datasetPath)
		if err != nil {
			t.Fatal(err)
		}
		construction, processor, err := bootstrapConstruction(oscillatorimage.Config{ModelType: "fixture"}, testMatrix.End, 1, declaration)
		if err != nil {
			t.Fatal(err)
		}
		session, err := scratchSession(t.Context(), store, construction, processor,
			generationContract(trainingprogram.ObjectiveLatentL2, recipecontract.ModalityText, recipecontract.ModalityImage, trainingprogram.MetricImagePSNRUnit))
		if err != nil {
			t.Fatal(err)
		}
		session.data = declaration
		plan, err := probeSession(trainingprogram.ObjectiveLatentL2, 3, testMatrix.End, 0)
		if err != nil {
			t.Fatal(err)
		}
		matrix := newHostMatrix(t, trainingprogram.ObjectiveLatentL2, plan.Optimizer())
		target := filepath.Join(root, "checkpoint")
		if err := (boundedRun{name: "scratch", plan: plan, session: session, loss: matrix.loss,
			step: func(int) (optimizer.ObservedStepResult, error) { return matrix.step(), nil },
			publish: func(ctx context.Context) error {
				return session.publishPlanned(ctx, target, matrix, plan.Updates())
			},
		}).train(t.Context()); err != nil {
			t.Fatal(err)
		}
		checkpoint, err := trainingprogram.LoadCheckpoint(target)
		if err != nil {
			t.Fatal(err)
		}
		scratch := trainingprogram.LineageParent{Artifact: construction.ID(), Relation: artifact.RelationDependsOn}
		if checkpoint.Model != construction.InitializedModel() || checkpoint.Program != matrix.program.ID() || !slices.Contains(checkpoint.Lineage, scratch) {
			t.Fatalf("scratch checkpoint of %s under program %s with lineage %v, want the latent-l2 program %s from construction %s",
				checkpoint.Model, checkpoint.Program, checkpoint.Lineage, matrix.program.ID(), construction.ID())
		}
	})
	t.Run("actual program and data", func(t *testing.T) {
		ran := runUnderContract(t, flow, trainingprogram.ObjectiveFlowMatching, noCancel)
		if ran.err != nil || ran.steps != 3 {
			t.Fatalf("flow-matching run = (%d steps, %v), want all three", ran.steps, ran.err)
		}
		checkpoint, err := trainingprogram.LoadCheckpoint(ran.target)
		if err != nil {
			t.Fatal(err)
		}
		program, err := trainingprogram.CompileObjectiveProgram(trainingprogram.ObjectiveFlowMatching, nil, newHostMatrix(t, trainingprogram.ObjectiveFlowMatching, checkpoint.Optimizer.Config).plan)
		if err != nil {
			t.Fatal(err)
		}
		stream, err := trainingworkflow.TextStream([]byte(routeFixtureDataset), uint64(ran.steps))
		if err != nil {
			t.Fatal(err)
		}
		session := ran.session
		_, recorded, err := session.store.Artifact(t.Context(), checkpoint.ID())
		if err != nil || !recorded || session.authority.Objective != trainingprogram.ObjectiveFlowMatching || checkpoint.Program != program.ID() ||
			checkpoint.Stream != trainingprogram.DatasetState(stream) || checkpoint.Model != session.model || *session.published != checkpoint.ID() {
			t.Fatalf("checkpoint of %s under program %s at %+v recorded %t (%v), want the flow-matching program %s at %+v of %s",
				checkpoint.Model, checkpoint.Program, checkpoint.Stream, recorded, err, program.ID(), stream, session.model)
		}
	})
	t.Run("undeclared objective", func(t *testing.T) {
		ran := runUnderContract(t, trainingworkflow.TokenContract, trainingprogram.ObjectiveLatentL2, noCancel)
		if ran.err == nil || !strings.Contains(ran.err.Error(), "steps latent-l2") || ran.steps != 0 {
			t.Fatalf("latent-l2 run under a token recipe = (%d steps, %v), want it refused before any step", ran.steps, ran.err)
		}
		if _, err := os.Stat(ran.target); !os.IsNotExist(err) {
			t.Fatalf("the refused run published at %s (%v)", ran.target, err)
		}
		if observation, err := ran.session.observer.Finish(t.Context(), ran.session.model, ran.session.recipe, nil, 0); err != nil || observation.Valid() {
			t.Fatalf("a second finish = (%s, %v), want the refused session finished once", observation, err)
		}
	})
	t.Run("cancelled", func(t *testing.T) {
		for name, steps := range map[string]int{"before training": 0, "after one update": 1} {
			ran := runUnderContract(t, flow, trainingprogram.ObjectiveFlowMatching, steps)
			if !errors.Is(ran.err, context.Canceled) || ran.steps != steps {
				t.Fatalf("a run cancelled %s = (%d steps, %v), want %d and its cancellation", name, ran.steps, ran.err, steps)
			}
			if _, err := os.Stat(ran.target); !os.IsNotExist(err) {
				t.Fatalf("a run cancelled %s published at %s (%v)", name, ran.target, err)
			}
		}
	})
	t.Run("setup failure", func(t *testing.T) {
		_, store, modelPath, datasetPath := routeFixture(t)
		session, err := openSession(t.Context(), store, modelPath, datasetPath)
		if err != nil {
			t.Fatal(err)
		}
		failure := errors.New("the model failed to load")
		setupErr := failure
		session.settle(t.Context(), &setupErr)
		if setupErr != failure {
			t.Fatalf("settling a failed setup = %v, want its own failure kept", setupErr)
		}
		if observation, err := session.observer.Finish(t.Context(), session.model, session.recipe, nil, 0); err != nil || observation.Valid() {
			t.Fatalf("a finish after settling = (%s, %v), want the session finished once", observation, err)
		}
	})
}

// noCancel lets a run under contract take every update.
const noCancel = -1

// authorityRun is what a run under contract left: its outcome, the steps
// it took, its session and where it publishes.
type authorityRun struct {
	err     error
	steps   int
	session routeSession
	target  string
}

// runUnderContract trains a host matrix stepping kind for three updates
// under a session opened with contract over a route fixture whose dataset
// file is edited once the session read it. Its context ends once the run
// has taken cancelAt steps, before training for zero, never for noCancel.
func runUnderContract(t *testing.T, contract trainingworkflow.RecipeContract, kind trainingprogram.ObjectiveKind, cancelAt int) authorityRun {
	t.Helper()
	root, store, modelPath, datasetPath := routeFixture(t)
	session, err := openSessionAs(t.Context(), store, modelPath, datasetPath, contract)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(datasetPath, []byte("edited after the session read it"), 0o600); err != nil {
		t.Fatal(err)
	}
	plan, err := probeSession(kind, 3, testMatrix.End, 0)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	if cancelAt == 0 {
		cancel()
	}
	matrix := newHostMatrix(t, kind, plan.Optimizer())
	ran := authorityRun{session: session, target: filepath.Join(root, "checkpoint")}
	ran.err = boundedRun{name: "authority", plan: plan, session: session, loss: matrix.loss,
		step: func(int) (optimizer.ObservedStepResult, error) {
			if ran.steps++; ran.steps == cancelAt {
				cancel()
			}
			return matrix.step(), nil
		},
		publish: func(ctx context.Context) error {
			return session.publishPlanned(ctx, ran.target, matrix, plan.Updates())
		},
	}.train(ctx)
	return ran
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
// trains under, as the forecast and table routes do, then holds the
// checkpoint to loading with that program and being recorded in the store.
func publishUnderObjective(t *testing.T, ctx context.Context, store *overgodb.Store, recipeID artifact.ID, kind trainingprogram.ObjectiveKind, stream artifact.ID) (trainingprogram.Checkpoint, error) {
	t.Helper()
	config, err := trainingprogram.BuiltinOptimizerPolicy().Config(testMatrix.End)
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	matrix := newHostMatrix(t, kind, config)
	result := matrix.step()
	target := filepath.Join(t.TempDir(), "checkpoint")
	published, err := publishTrained(ctx, store, recipeID, target, matrix, trainingdata.StreamState{Identity: stream, Position: uint64(result.Step)})
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	loaded, err := trainingprogram.LoadCheckpoint(target)
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	if _, recorded, err := store.Artifact(ctx, loaded.ID()); err != nil || !recorded || loaded.Program != matrix.program.ID() || loaded.ID() != published.ID() {
		t.Fatalf("checkpoint %s of program %s recorded %t (%v), want %s's checkpoint %s recorded", loaded.ID(), loaded.Program, recorded, err, matrix.program.ID(), published.ID())
	}
	return published, nil
}
