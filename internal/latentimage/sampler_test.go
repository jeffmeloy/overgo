package latentimage

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// g2 schedule golden shape (only the fields this test asserts).
type g2Schedule struct {
	Schedule struct {
		Mu                float64   `json:"mu"`
		NumTrainTimesteps int       `json:"num_train_timesteps"`
		Scheduler         string    `json:"scheduler"`
		Steps             int       `json:"steps"`
		SigmaKnots        []float64 `json:"sigma_knots"`
		SigmasCurrent     []float64 `json:"sigmas_current"`
		SigmasNext        []float64 `json:"sigmas_next"`
		SigmaDeltas       []float64 `json:"sigma_deltas"`
		Timesteps         []float64 `json:"timesteps"`
	} `json:"schedule"`
}

func loadG2(t *testing.T) g2Schedule {
	t.Helper()
	path := filepath.Join("..", "..", "fixtures", "krea", "g2_schedule.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read g2 golden: %v", err)
	}
	var g g2Schedule
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse g2 golden: %v", err)
	}
	return g
}

// The sampler schedule must reproduce the g2 knots EXACTLY from config (mu,
// steps, N) with no hardcoded knots -- shift = exp(mu), sigma =
// shiftFlowSigma(1-i/steps, shift), t = sigma*N.
func TestFlowScheduleReproducesG2Exactly(t *testing.T) {
	g := loadG2(t)
	if g.Schedule.Scheduler != "FlowMatchEulerDiscreteScheduler" {
		t.Fatalf("unexpected scheduler %q", g.Schedule.Scheduler)
	}
	sched, err := CompileFlowSchedule(g.Schedule.Steps, g.Schedule.NumTrainTimesteps, g.Schedule.Mu)
	if err != nil {
		t.Fatalf("CompileFlowSchedule: %v", err)
	}

	// sigma_knots: Steps+1 entries, descending to terminal 0.
	if len(sched.Sigmas) != len(g.Schedule.SigmaKnots) {
		t.Fatalf("sigma count %d != golden %d", len(sched.Sigmas), len(g.Schedule.SigmaKnots))
	}
	for i, want := range g.Schedule.SigmaKnots {
		if sched.Sigmas[i] != want {
			t.Errorf("sigma_knot[%d] = %.17g, want %.17g (ulp %d)",
				i, sched.Sigmas[i], want, ulpDistance(sched.Sigmas[i], want))
		}
	}
	// terminal knot exactly zero.
	if sched.Sigmas[len(sched.Sigmas)-1] != 0 {
		t.Errorf("terminal sigma = %.17g, want 0", sched.Sigmas[len(sched.Sigmas)-1])
	}
	// sigmas_current / sigmas_next views.
	for i, want := range g.Schedule.SigmasCurrent {
		if sched.Sigmas[i] != want {
			t.Errorf("sigma_current[%d] = %.17g, want %.17g", i, sched.Sigmas[i], want)
		}
	}
	for i, want := range g.Schedule.SigmasNext {
		if sched.Sigmas[i+1] != want {
			t.Errorf("sigma_next[%d] = %.17g, want %.17g", i, sched.Sigmas[i+1], want)
		}
	}
	// timesteps: sigma*N.
	if len(sched.Timesteps) != len(g.Schedule.Timesteps) {
		t.Fatalf("timestep count %d != golden %d", len(sched.Timesteps), len(g.Schedule.Timesteps))
	}
	for i, want := range g.Schedule.Timesteps {
		if sched.Timesteps[i] != want {
			t.Errorf("timestep[%d] = %.17g, want %.17g (ulp %d)",
				i, sched.Timesteps[i], want, ulpDistance(sched.Timesteps[i], want))
		}
	}
	// sigma_deltas: Sigmas[i+1]-Sigmas[i].
	for i, want := range g.Schedule.SigmaDeltas {
		if sched.Deltas[i] != want {
			t.Errorf("sigma_delta[%d] = %.17g, want %.17g (ulp %d)",
				i, sched.Deltas[i], want, ulpDistance(sched.Deltas[i], want))
		}
	}
}

// The shift is derived from mu (exp(mu)); assert the derivation is not a
// hardcoded literal by cross-checking a second mu value round-trips.
func TestDynamicShiftDerivation(t *testing.T) {
	g := loadG2(t)
	if got := DynamicShift(g.Schedule.Mu); got != math.Exp(g.Schedule.Mu) {
		t.Fatalf("DynamicShift(%g) = %g, want exp = %g", g.Schedule.Mu, got, math.Exp(g.Schedule.Mu))
	}
	// A different seq-len-derived mu must yield a different schedule (no
	// hardcoded knots): mu=0.5 (the base_shift end of the dynamic map).
	other, err := CompileFlowSchedule(g.Schedule.Steps, g.Schedule.NumTrainTimesteps, 0.5)
	if err != nil {
		t.Fatalf("CompileFlowSchedule(mu=0.5): %v", err)
	}
	if other.Sigmas[1] == g.Schedule.SigmaKnots[1] {
		t.Fatalf("mu=0.5 reproduced the mu=1.15 knot; schedule is hardcoded")
	}
}

func TestEulerStepAdvancesBySigmaDelta(t *testing.T) {
	sched, err := CompileFlowSchedule(8, 1000, 1.15)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	sample := []float64{1, 2, 3}
	velocity := []float64{0.5, -0.5, 1}
	before := append([]float64(nil), sample...)
	if err := sched.EulerStep(sample, velocity, 0); err != nil {
		t.Fatalf("euler step: %v", err)
	}
	for i := range sample {
		want := before[i] + sched.Deltas[0]*velocity[i]
		if sample[i] != want {
			t.Errorf("euler[%d] = %g, want %g", i, sample[i], want)
		}
	}
}

// ulpDistance reports the count of representable doubles between a and b (for
// diagnostics when an exact compare fails).
func ulpDistance(a, b float64) int64 {
	if a == b {
		return 0
	}
	ai := int64(math.Float64bits(a))
	bi := int64(math.Float64bits(b))
	if ai < 0 {
		ai = math.MinInt64 - ai
	}
	if bi < 0 {
		bi = math.MinInt64 - bi
	}
	d := ai - bi
	if d < 0 {
		d = -d
	}
	return d
}
