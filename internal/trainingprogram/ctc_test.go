package trainingprogram

import (
	"context"
	"errors"
	"math"
	"os"
	"slices"
	"sync/atomic"
	"testing"

	"overgo/internal/optimizer"
	"overgo/internal/recipecontract"
	"overgo/internal/strictjson"
	"overgo/internal/testutil"
)

type ctcOracleCase struct {
	Name       string    `json:"name"`
	Frames     int       `json:"frames"`
	Vocabulary int       `json:"vocabulary"`
	Blank      int       `json:"blank"`
	Targets    []int     `json:"targets"`
	Logits     []float32 `json:"logits"`
	Loss       *float64  `json:"loss,omitempty"`
	Gradient   []float64 `json:"gradient,omitempty"`
	Refusal    string    `json:"refusal,omitzero"`
}

func ctcOracleCases(t *testing.T) []ctcOracleCase {
	t.Helper()
	data, err := os.ReadFile("testdata/ctc_oracle.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema        string          `json:"schema"`
		TorchVersion  string          `json:"torch_version"`
		TorchRevision string          `json:"torch_revision"`
		Device        string          `json:"device"`
		DType         string          `json:"dtype"`
		Reduction     string          `json:"reduction"`
		ZeroInfinity  bool            `json:"zero_infinity"`
		Source        string          `json:"source"`
		Cases         []ctcOracleCase `json:"cases"`
	}
	if err := strictjson.DecodeBytes(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != "overgo/ctc-objective-oracle/v1" || fixture.Device != "cpu" || fixture.DType != "float64-from-float32" ||
		fixture.Reduction != "sum" || fixture.ZeroInfinity || fixture.TorchVersion != "2.12.0+cu126" ||
		fixture.TorchRevision != "7661cd9c6b841b62b7f411aa52ec51f05457263b" ||
		fixture.Source != "https://github.com/pytorch/pytorch/blob/"+fixture.TorchRevision+"/aten/src/ATen/native/LossCTC.cpp" {
		t.Fatal("CTC oracle provenance or loss semantics changed")
	}
	required := []string{"one_frame", "empty_target", "multiple_paths", "repeated_minimum", "repeated_many_paths", "nonzero_blank", "alternating_targets", "many_alignments", "large_logits", "common_offset", "single_class_empty", "impossible_repeat", "impossible_length"}
	if len(fixture.Cases) != len(required) {
		t.Fatalf("oracle cases = %d, want %d", len(fixture.Cases), len(required))
	}
	for index, test := range fixture.Cases {
		if test.Name != required[index] || test.Frames <= 0 || test.Vocabulary <= 0 || len(test.Logits) != test.Frames*test.Vocabulary {
			t.Fatalf("invalid oracle case %d: %+v", index, test)
		}
		if test.Refusal == "" && (test.Loss == nil || !finite(*test.Loss) || len(test.Gradient) != len(test.Logits)) {
			t.Fatalf("oracle case %s lacks finite loss/gradient evidence", test.Name)
		}
		if test.Refusal != "" && (test.Refusal != "impossible-alignment" || test.Loss != nil || test.Gradient != nil) {
			t.Fatalf("invalid oracle refusal %s", test.Name)
		}
	}
	return fixture.Cases
}

func ctcScratch(t testing.TB, test ctcOracleCase) []float64 {
	t.Helper()
	count, err := CTCLossWorkspaceSize(test.Frames, test.Vocabulary, len(test.Targets))
	if err != nil {
		t.Fatal(err)
	}
	return make([]float64, count)
}

// TestCTCObjectiveAcceptance is the CPU objective's complete acceptance. The
// named children are independently runnable development checkpoints, not an
// ASR training, dataset-quality or device-performance claim.
func TestCTCObjectiveAcceptance(t *testing.T) {
	cases := ctcOracleCases(t)
	t.Run("reference", func(t *testing.T) {
		for _, test := range cases {
			if test.Refusal != "" {
				continue // Refusals are mandatory in the admission checkpoint.
			}
			t.Run(test.Name, func(t *testing.T) {
				workspace := ctcScratch(t, test)
				gradient := make([]float32, len(test.Logits))
				original := slices.Clone(test.Logits)
				loss, err := CTCLossF32(t.Context(), gradient, test.Logits, test.Targets, test.Frames, test.Vocabulary, test.Blank, workspace)
				if err != nil {
					t.Fatal(err)
				}
				// Float64 accumulation allowance scales with the DP geometry and
				// loss magnitude; gradient comparison below allows one float32 ULP at 1.
				tolerance := 32 * (math.Nextafter(1, 2) - 1) * float64(len(workspace)) * max(1, math.Abs(*test.Loss))
				testutil.RequireWithin(t, "CTC loss", []float64{loss}, []float64{*test.Loss}, tolerance)
				withoutGradient, err := CTCLossF32(t.Context(), nil, test.Logits, test.Targets, test.Frames, test.Vocabulary, test.Blank, workspace)
				if err != nil || withoutGradient != loss || !slices.Equal(test.Logits, original) {
					t.Fatalf("loss-only/input preservation = %g, %v", withoutGradient, err)
				}
			})
		}
		t.Run("compiled_objective", func(t *testing.T) {
			for _, input := range []recipecontract.Modality{recipecontract.ModalityAudio, recipecontract.ModalityImage} {
				spec := objectiveFixture(t, input, recipecontract.ModalityText, false).ObjectiveSpec
				spec.Kind = ObjectiveCTC
				if _, err := NewObjective(spec); err != nil {
					t.Fatal(err)
				}
			}
			plan, err := optimizer.CompilePlan(1, []optimizer.GroupSpec{{Name: "weight", End: 1, Rows: 1, Cols: 1}})
			if err != nil {
				t.Fatal(err)
			}
			program, err := CompileObjectiveProgram(ObjectiveCTC, nil, plan)
			if err != nil || program.Objective() != ObjectiveCTC {
				t.Fatalf("CTC program = %+v, %v", program, err)
			}
		})
	})
	t.Run("gradients", func(t *testing.T) {
		for _, test := range cases {
			if test.Refusal != "" {
				continue
			}
			t.Run(test.Name, func(t *testing.T) {
				workspace := ctcScratch(t, test)
				gradient := make([]float32, len(test.Logits))
				loss, err := CTCLossF32(t.Context(), gradient, test.Logits, test.Targets, test.Frames, test.Vocabulary, test.Blank, workspace)
				if err != nil {
					t.Fatal(err)
				}
				epsilon := float64(math.Nextafter32(1, 2) - 1)
				testutil.RequireElementsWithin(t, "CTC logits gradient", gradient, test.Gradient, epsilon)
				probe := slices.Clone(test.Logits)
				for index, value := range probe {
					// An exact binary perturbation, enlarged only when float32
					// spacing requires it. Use the actual representable span.
					step := max(float32(1.0/1024), 2*(math.Nextafter32(value, float32(math.Inf(1)))-value))
					probe[index] = value + step
					upper := probe[index]
					plus, err := CTCLossF32(t.Context(), nil, probe, test.Targets, test.Frames, test.Vocabulary, test.Blank, workspace)
					if err != nil {
						t.Fatal(err)
					}
					probe[index] = value - step
					span := float64(upper) - float64(probe[index])
					minus, err := CTCLossF32(t.Context(), nil, probe, test.Targets, test.Frames, test.Vocabulary, test.Blank, workspace)
					if err != nil || span <= 0 {
						t.Fatalf("finite difference = %v, span %g", err, span)
					}
					probe[index] = value
					// A single-logit third derivative is a difference of Bernoulli
					// third cumulants (absolute bound < 1). Add float64 cancellation
					// and the final float32 rounding allowance.
					tolerance := span*span/24 + 64*(math.Nextafter(1, 2)-1)*float64(len(workspace))*max(1, math.Abs(loss))/span + epsilon
					testutil.RequireWithin(t, "finite difference", []float32{gradient[index]}, []float64{(plus - minus) / span}, tolerance)
				}
				for frame := range test.Frames {
					var sum float64
					for _, value := range gradient[frame*test.Vocabulary : (frame+1)*test.Vocabulary] {
						sum += float64(value)
					}
					if math.Abs(sum) > float64(test.Vocabulary)*epsilon {
						t.Fatalf("frame %d gradient sums to %g", frame, sum)
					}
				}
			})
		}
	})
	t.Run("admission", func(t *testing.T) {
		for _, test := range cases {
			if test.Refusal == "" {
				continue
			}
			t.Run(test.Name, func(t *testing.T) {
				if _, err := CTCLossF32(t.Context(), make([]float32, len(test.Logits)), test.Logits, test.Targets, test.Frames, test.Vocabulary, test.Blank, ctcScratch(t, test)); err == nil {
					t.Fatal("impossible alignment became a successful loss")
				}
			})
		}
		ctcAdmission(t, cases[2])
	})
}

func ctcAdmission(t *testing.T, test ctcOracleCase) {
	t.Helper()
	for _, malformed := range []struct {
		name   string
		mutate func(*ctcOracleCase)
	}{
		{"zero_frames", func(c *ctcOracleCase) { c.Frames = 0 }},
		{"negative_frames", func(c *ctcOracleCase) { c.Frames = -1 }},
		{"zero_vocabulary", func(c *ctcOracleCase) { c.Vocabulary = 0 }},
		{"shape_overflow", func(c *ctcOracleCase) { c.Frames = math.MaxInt }},
		{"short_logits", func(c *ctcOracleCase) { c.Logits = c.Logits[:len(c.Logits)-1] }},
		{"negative_blank", func(c *ctcOracleCase) { c.Blank = -1 }},
		{"blank_out_of_range", func(c *ctcOracleCase) { c.Blank = c.Vocabulary }},
		{"target_blank", func(c *ctcOracleCase) { c.Targets = []int{c.Blank} }},
		{"target_negative", func(c *ctcOracleCase) { c.Targets = []int{-1} }},
		{"target_out_of_range", func(c *ctcOracleCase) { c.Targets = []int{c.Vocabulary} }},
		{"nan", func(c *ctcOracleCase) { c.Logits[0] = float32(math.NaN()) }},
		{"positive_infinity", func(c *ctcOracleCase) { c.Logits[0] = float32(math.Inf(1)) }},
		{"negative_infinity", func(c *ctcOracleCase) { c.Logits[0] = float32(math.Inf(-1)) }},
	} {
		t.Run(malformed.name, func(t *testing.T) {
			candidate := test
			candidate.Logits = slices.Clone(test.Logits)
			malformed.mutate(&candidate)
			if _, err := CTCLossF32(t.Context(), nil, candidate.Logits, candidate.Targets, candidate.Frames, candidate.Vocabulary, candidate.Blank, ctcScratch(t, test)); err == nil {
				t.Fatal("malformed CTC objective admitted")
			}
		})
	}
	t.Run("storage", func(t *testing.T) {
		workspace := ctcScratch(t, test)
		gradient := make([]float32, len(test.Logits))
		call := func(ctx context.Context, dst, logits []float32, scratch []float64) error {
			_, err := CTCLossF32(ctx, dst, logits, test.Targets, test.Frames, test.Vocabulary, test.Blank, scratch)
			return err
		}
		if call(nil, gradient, test.Logits, workspace) == nil || call(t.Context(), gradient[:len(gradient)-1], test.Logits, workspace) == nil ||
			call(t.Context(), gradient, test.Logits, workspace[:len(workspace)-1]) == nil || call(t.Context(), test.Logits, test.Logits, workspace) == nil {
			t.Fatal("nil context, short storage or aliased gradient admitted")
		}
		backing := append(slices.Clone(test.Logits), 0)
		if call(t.Context(), backing[1:], backing[:len(test.Logits)], workspace) == nil {
			t.Fatal("partially overlapping gradient admitted")
		}
		for _, geometry := range [][3]int{{0, 3, 1}, {1, 0, 1}, {1, 3, -1}, {1, 3, math.MaxInt}, {math.MaxInt, 3, 1}, {math.MaxInt / 2, 3, 2}, {1, math.MaxInt / 8, 0}} {
			if _, err := CTCLossWorkspaceSize(geometry[0], geometry[1], geometry[2]); err == nil {
				t.Fatalf("invalid/overflowed workspace admitted: %v", geometry)
			}
		}
		if allocations := testing.AllocsPerRun(100, func() {
			if err := call(t.Context(), gradient, test.Logits, workspace); err != nil {
				t.Fatal(err)
			}
		}); allocations != 0 {
			t.Fatalf("CTC allocated beyond caller-owned storage: %g", allocations)
		}
	})
	t.Run("cancellation", func(t *testing.T) {
		for _, checks := range []int{1, 2, test.Frames + 2, 2*test.Frames + 2} {
			ctx, cancel := context.WithCancelCause(t.Context())
			controlled := new(ctcCancelContext{Context: ctx, cancel: cancel})
			controlled.remaining.Store(int64(checks))
			_, err := CTCLossF32(controlled, make([]float32, len(test.Logits)), test.Logits, test.Targets, test.Frames, test.Vocabulary, test.Blank, ctcScratch(t, test))
			cancel(nil)
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("cancellation at context check %d = %v", checks, err)
			}
		}
	})
}

// ctcCancelContext cancels its actual underlying context at a deterministic
// check, covering normalization, forward and backward without timer races.
type ctcCancelContext struct {
	context.Context
	cancel    context.CancelCauseFunc
	remaining atomic.Int64
}

func (c *ctcCancelContext) Err() error {
	if c.remaining.Add(-1) == 0 {
		c.cancel(context.Canceled)
	}
	return c.Context.Err()
}
