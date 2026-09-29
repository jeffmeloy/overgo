package hybridtrain

import (
	"math"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/optimizer"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
)

const (
	testSeed         = int64(20250812)
	testSteps        = 8
	testLearningRate = 0.02
	testMomentum     = 0.9
)

// TestResidentMatrixLayout verifies exact resident partitioning.
func TestResidentMatrixLayout(t *testing.T) {
	cfg := smallHybrid()
	m, err := BuildModel(cfg, testSeed)
	if err != nil {
		t.Fatal(err)
	}
	plans, err := m.matrixPlans()
	if err != nil {
		t.Fatal(err)
	}
	if len(plans) != len(cfg.Types) {
		t.Fatalf("plans=%d, want %d layers", len(plans), len(cfg.Types))
	}
	if err := validateMatrixTiling(plans, m.MatrixParamCount()); err != nil {
		t.Fatalf("matrix tiling invalid: %v", err)
	}

	// Range geometry.
	byName := map[string]matDesc{}
	for _, md := range m.mats {
		byName[md.name] = md
	}
	total := 0
	for _, md := range m.mats {
		if md.size != md.rows*md.cols {
			t.Errorf("matrix %q size=%d != rows*cols=%d", md.name, md.size, md.rows*md.cols)
		}
		total += md.size
	}
	if total != m.MatrixParamCount() {
		t.Errorf("sum of matrix sizes=%d != matW len=%d", total, m.MatrixParamCount())
	}

	// Complete coverage.
	mapped := 0
	for _, p := range plans {
		mapped += len(p.slots())
	}
	if mapped != len(m.mats) {
		t.Errorf("plans map %d matrices, but model has %d", mapped, len(m.mats))
	}
	t.Logf("layout OK: %d layers, %d matrices tiling %d matW elements", len(plans), len(m.mats), m.MatrixParamCount())
	if !m.Program().ID().Valid() {
		t.Fatal("compiled program absent")
	}
}

// TestHybridCompiledHostLifecycle trains the host lane to a finite decrease
// and holds the checkpoint it ends at to reloading: every trainable group's
// saved weights read back bit-exact beside momentum covering them all.
func TestHybridCompiledHostLifecycle(t *testing.T) {
	model, err := BuildModel(smallHybrid(), testSeed)
	if err != nil {
		t.Fatal(err)
	}
	var state optimizer.State
	trajectory, err := model.TrainHost(testSteps, optimizer.Config{
		BaseLearningRate: testLearningRate, Momentum: testMomentum, Schedule: optimizer.ScheduleConstant,
	}, TrainingOptions{Checkpoint: func(snapshot optimizer.State) error { state = snapshot; return nil }})
	if err != nil {
		t.Fatal(err)
	}
	testutil.RequireFiniteDecrease(t, "hybrid host", trajectory, testSteps)
	directory := t.TempDir()
	if err := model.SaveWeights(filepath.Join(directory, "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	groups := slices.Concat(model.matPlan.Groups(), model.vecPlan.Groups())
	if len(source.Tensors) != len(groups) || len(state.Momentum) != len(model.matW)+len(model.vecW) || state.Step != testSteps {
		t.Fatalf("checkpoint holds %d tensors and %d momentum at step %d, want %d, %d and %d",
			len(source.Tensors), len(state.Momentum), state.Step, len(groups), len(model.matW)+len(model.vecW), testSteps)
	}
	for index, group := range groups {
		trained := model.matW
		if index >= model.matPlan.GroupCount() {
			trained = model.vecW
		}
		loaded, err := safetensors.ReadF32(source.Tensors[group.Name])
		if err != nil || !slices.EqualFunc(loaded, trained[group.Start:group.End], func(read, written float32) bool {
			return math.Float32bits(read) == math.Float32bits(written)
		}) {
			t.Fatalf("tensor %s reads back (%d values, %v), want its %d trained weights bit-exact", group.Name, len(loaded), err, group.End-group.Start)
		}
	}
}
