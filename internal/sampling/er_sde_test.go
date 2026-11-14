package sampling

import (
	"math"
	"testing"

	"overgo/internal/checked"
	"overgo/internal/jsonfile"
	"overgo/internal/testcheck"
)

// comfyERSDE is ComfyUI's er_sde run over a flow model's sigmas with a stand-in model, whose data prediction is the
// target plus half of (1 − σ) times the latent (scripts/python_golden_references/comfy_er_sde_golden.py): each case's
// first latent, its target, every fresh noise draw it took, the latent at each step and the result.
type comfyERSDE struct {
	Cases []struct {
		Name       string      `json:"name"`
		Sigmas     []float32   `json:"sigmas"`
		Draws      int         `json:"draws"`
		X0         []float32   `json:"x0"`
		Target     []float32   `json:"target"`
		Noise      [][]float32 `json:"noise"`
		Trajectory [][]float32 `json:"trajectory"`
		Result     []float32   `json:"result"`
	} `json:"cases"`
}

// erSDETolerance is how far a latent may lie from ComfyUI's: float32 rounding, at which 200 integration points land
// (5e-7) and the paper's 100 do not (3e-4).
const erSDETolerance = 5e-6

// TestERSDEDrawsNoiseAsTorchDoes holds ER-SDE (moss-ultra-sections/er-sde), written from arXiv 2309.06169, to ComfyUI's
// er_sde over eight steps of bong_tangent from σ = 1, of beta57, and of bong_tangent at denoise 0.72: it draws fresh
// noise at the steps ComfyUI drew it, every step but the last, and given the same draws its latent at every step and its
// result lie within float32 rounding of ComfyUI's.
func TestERSDEDrawsNoiseAsTorchDoes(t *testing.T) {
	var golden comfyERSDE
	testcheck.NoError(t, jsonfile.Decode("testdata/comfy_er_sde_trajectories.json", &golden))
	worst := erSDEOff
	for _, c := range golden.Cases {
		solver := testcheck.Must(NewERSDE(c.Sigmas, ERSDEStages))(t)
		x, drawn := c.X0, 0
		for step := range solver.Steps() {
			if off := worst(x, c.Trajectory[step]); off > erSDETolerance {
				t.Errorf("%s step %d: the latent is %.3g off ComfyUI's", c.Name, step, off)
			}
			denoised := make([]float32, len(x))
			for i := range x {
				denoised[i] = c.Target[i] + 0.5*(1-c.Sigmas[step])*x[i]
			}
			var noise []float32
			if solver.Draws(step) {
				noise = c.Noise[drawn]
				drawn++
			}
			x = testcheck.Must(solver.Step(x, denoised, noise))(t)
		}
		if off := worst(x, c.Result); off > erSDETolerance || drawn != c.Draws {
			t.Errorf("%s: the result is %.3g off ComfyUI's after %d draws, ComfyUI's %d", c.Name, off, drawn, c.Draws)
		}
	}
	if len(golden.Cases) < 3 {
		t.Errorf("the golden holds %d cases", len(golden.Cases))
	}
}

// erSDEOff is how far got lies from want at its farthest element, infinitely far where a difference is not finite: a
// NaN is no number max can rank, and a NaN off compared to the tolerance reads as agreement (SQA-396).
func erSDEOff(got, want []float32) float64 {
	off := 0.0
	for i := range want {
		difference := math.Abs(float64(got[i]) - float64(want[i]))
		if !checked.Finite64(difference) {
			return math.Inf(1)
		}
		off = max(off, difference)
	}
	return off
}

// TestTheERSDEComparisonRefusesANonfiniteError holds the comparison ER-SDE's parity reads (er-sde-finite/do, SQA-396)
// to a refusal where the latent is not finite: a NaN or an infinity beside a finite reference lies past the tolerance,
// as a latent within float32 rounding of it does not.
func TestTheERSDEComparisonRefusesANonfiniteError(t *testing.T) {
	want := []float32{0.25, -0.5, 1}
	for name, got := range map[string][]float32{
		"NaN":      {0.25, float32(math.NaN()), 1},
		"infinity": {0.25, -0.5, float32(math.Inf(1))},
	} {
		if off := erSDEOff(got, want); !(off > erSDETolerance) {
			t.Errorf("a latent holding a %s reads %g off the reference, within the tolerance", name, off)
		}
	}
	if off := erSDEOff([]float32{0.25, -0.5, 1.000001}, want); off > erSDETolerance {
		t.Errorf("a latent within float32 rounding of the reference reads %g off it", off)
	}
}

// TestERSDETakesItsStepsInOrder holds the solver's bounds: a step with noise where none is drawn, or past the last.
func TestERSDETakesItsStepsInOrder(t *testing.T) {
	solver := testcheck.Must(NewERSDE([]float32{1, 0.5, 0}, ERSDEStages))(t)
	x := []float32{1, 2}
	if _, err := solver.Step(x, x, nil); err == nil {
		t.Error("a drawing step was taken without noise")
	}
	x = testcheck.Must(solver.Step(x, x, []float32{0, 0}))(t)
	x = testcheck.Must(solver.Step(x, x, nil))(t)
	if _, err := solver.Step(x, x, nil); err == nil {
		t.Error("a step past the last was taken")
	}
	if _, err := NewERSDE([]float32{1, 0}, 4); err == nil {
		t.Error("four stages were taken")
	}
}
