package hybridtrain

import (
	"testing"

	"overgo/internal/optimizer"
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

func TestHybridCompiledHostLifecycle(t *testing.T) {
	model, err := BuildModel(smallHybrid(), testSeed)
	if err != nil {
		t.Fatal(err)
	}
	trajectory, err := model.TrainHost(testSteps, optimizer.Config{
		BaseLearningRate: testLearningRate, Momentum: testMomentum, Schedule: optimizer.ScheduleConstant,
	})
	if err != nil {
		t.Fatal(err)
	}
	testutil.RequireFiniteDecrease(t, "hybrid host", trajectory, testSteps)
}
