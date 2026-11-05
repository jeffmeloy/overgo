package hybridtrain

import (
	"math"
	"slices"
	"testing"

	"overgo/internal/checkpointtest"
	"overgo/internal/optimizer"
	"overgo/internal/testcheck"
)

// checkpointed is a hybrid model with the optimizer state its checkpoint holds, as a trainer hands both on.
type checkpointed struct {
	*Model
	state optimizer.State
}

func (c checkpointed) Snapshot() optimizer.State { return c.state }

// TestAHybridCheckpointKeepsMatrixMomentumInF32: a device stepper's checkpoint holds the matrix momentum in the f32 it is
// stepped in, the stepper's own slab with no f64 copy of it, then the vector slab's f64 momentum. It is published and
// read back bit for bit, and a resume continues from the same f32 matrix momentum and f64 vector momentum. A checkpoint
// written before, all f64, resumes to the same, and a host stepper's resume widens the f32 exactly.
func TestAHybridCheckpointKeepsMatrixMomentumInF32(t *testing.T) {
	t.Parallel()
	m := progressModel(t)
	cfg := optimizer.Config{BaseLearningRate: testLearningRate, Momentum: testMomentum}
	const step = 3
	matrix := make([]float32, len(m.matW))
	for index := range matrix {
		matrix[index] = float32(index+1) / 7
	}
	vector := make([]float64, len(m.vecW))
	for index := range vector {
		vector[index] = float64(index+1) / 3
	}
	vec, err := optimizer.New(m.vecW, make([]float32, len(m.vecW)), m.vecPlan, cfg)
	testcheck.NoError(t, err)
	testcheck.NoError(t, vec.Restore(optimizer.State{PlanIdentity: m.vecPlan.Identity(), Config: cfg, Step: step, Momentum: vector}))
	var state optimizer.State
	testcheck.NoError(t, m.checkpointF32(TrainingOptions{Checkpoint: func(s optimizer.State) error { state = s; return nil }}, matrix, step, cfg, vec))
	if len(state.MomentumF32) != len(matrix) || &state.MomentumF32[0] != &matrix[0] || !slices.Equal(state.Momentum, vector) {
		t.Fatalf("the checkpoint holds %d f32 and %d f64 values, want the stepper's own %d f32 slab then the vector's %d f64",
			len(state.MomentumF32), len(state.Momentum), len(matrix), len(vector))
	}
	trained := map[string][]float32{}
	for _, slab := range []struct {
		plan   optimizer.Plan
		values []float32
	}{{m.matPlan, m.matW}, {m.vecPlan, m.vecW}} {
		for _, group := range slab.plan.Groups() {
			trained[group.Name] = slab.values[group.Start:group.End]
		}
	}
	checkpointtest.RequireRoundTrip(t, m.program, checkpointed{m, state}, trained)

	earlier := optimizer.State{PlanIdentity: state.PlanIdentity, Config: cfg, Step: step}
	for _, value := range matrix {
		earlier.Momentum = append(earlier.Momentum, float64(value))
	}
	earlier.Momentum = append(earlier.Momentum, vector...)
	for name, resume := range map[string]optimizer.State{"f32 then f64": state, "all f64, as written before": earlier} {
		mat, vec, err := m.resumeParts(1, cfg, TrainingOptions{Resume: &resume}, true)
		if err != nil || mat.Momentum != nil || !slices.Equal(mat.MomentumF32, matrix) || !slices.Equal(vec.Momentum, vector) || vec.MomentumF32 != nil {
			t.Fatalf("%s: a device resume holds %d f64 and %d f32 matrix values and %d vector values (%v), want the matrix's f32 and the vector's f64",
				name, len(mat.Momentum), len(mat.MomentumF32), len(vec.Momentum), err)
		}
		host, _, err := m.resumeParts(1, cfg, TrainingOptions{Resume: &resume}, false)
		if err != nil || host.MomentumF32 != nil || !slices.EqualFunc(host.Momentum, matrix, func(wide float64, narrow float32) bool { return wide == float64(narrow) }) {
			t.Fatalf("%s: a host resume did not widen the matrix momentum exactly: %v", name, err)
		}
	}
	// An f64 value the f32 a device steps cannot hold is refused, never rounded.
	earlier.Momentum[0] = math.Pi
	if _, _, err := m.resumeParts(1, cfg, TrainingOptions{Resume: &earlier}, true); err == nil {
		t.Fatal("a device resume rounded an f64 matrix momentum it cannot hold")
	}
}
