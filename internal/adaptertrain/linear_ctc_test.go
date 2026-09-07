package adaptertrain

import (
	"context"
	"errors"
	"math"
	"os"
	"slices"
	"testing"

	"overgo/internal/optimizer"
	"overgo/internal/strictjson"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

type linearCTCOracleCase struct {
	Name               string    `json:"name"`
	Frames             int       `json:"frames"`
	Width              int       `json:"width"`
	Vocabulary         int       `json:"vocabulary"`
	Blank              int       `json:"blank"`
	Targets            []int     `json:"targets"`
	Hidden             []float32 `json:"hidden"`
	OutputWeight       []float32 `json:"output_weight"`
	OutputBias         []float32 `json:"output_bias"`
	Projection         []float32 `json:"projection"`
	Logits             []float64 `json:"logits"`
	Loss               float64   `json:"loss"`
	ProjectionGradient []float64 `json:"projection_gradient"`
	LogitsGradient     []float64 `json:"logits_gradient"`
}

func linearCTCOracle(t *testing.T) []linearCTCOracleCase {
	t.Helper()
	data, err := os.ReadFile("testdata/linear_ctc_oracle.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema          string                `json:"schema"`
		Revision        string                `json:"torch_revision"`
		Device          string                `json:"device"`
		DType           string                `json:"dtype"`
		Reduction       string                `json:"reduction"`
		GeneratorSHA256 string                `json:"generator_sha256"`
		Cases           []linearCTCOracleCase `json:"cases"`
	}
	if err := strictjson.DecodeBytes(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != "overgo/linear-ctc-oracle/v1" || fixture.Revision != "7661cd9c6b841b62b7f411aa52ec51f05457263b" ||
		fixture.Device != "cpu" || fixture.DType != "float64-from-float32" || fixture.Reduction != "sum" ||
		fixture.GeneratorSHA256 != "b59580cbf005ecbdb590accd638aafedd192be933a265e6e44cb4ad03f50c33f" {
		t.Fatal("projection oracle provenance differs")
	}
	names := []string{"identity", "mixed_projection", "repeated_target", "empty_target", "nonzero_blank"}
	if len(fixture.Cases) != len(names) {
		t.Fatal("projection oracle coverage is incomplete")
	}
	for index, tc := range fixture.Cases {
		if tc.Name != names[index] || len(tc.ProjectionGradient) != tc.Width*tc.Width || len(tc.LogitsGradient) != tc.Frames*tc.Vocabulary ||
			!finiteProjectionValues(tc.Hidden) || math.IsNaN(tc.Loss) || math.IsInf(tc.Loss, 0) {
			t.Fatalf("invalid oracle case %s", tc.Name)
		}
	}
	return fixture.Cases
}

func linearCTCSpec(tc linearCTCOracleCase) LinearCTCSpec {
	return LinearCTCSpec{
		OutputWeight: tc.OutputWeight, OutputBias: tc.OutputBias, Width: tc.Width, Vocabulary: tc.Vocabulary,
		MaxFrames: tc.Frames, MaxTargets: len(tc.Targets), Blank: tc.Blank, MemoryBytes: 1 << 20,
		Optimizer: optimizer.Config{BaseLearningRate: 1e-3, Momentum: 0.9, Schedule: optimizer.ScheduleConstant},
	}
}

func TestLinearCTCReference(t *testing.T) {
	for _, tc := range linearCTCOracle(t) {
		t.Run(tc.Name, func(t *testing.T) {
			m, err := NewLinearCTC(linearCTCSpec(tc))
			if err != nil {
				t.Fatal(err)
			}
			copy(m.weights, tc.Projection)
			frozenWeight, frozenBias := slices.Clone(tc.OutputWeight), slices.Clone(tc.OutputBias)
			execution, err := m.Bind(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			gradients, err := execution.Select(trainingprogram.PhaseForward, trainingprogram.PhaseBackward)
			if err != nil {
				t.Fatal(err)
			}
			forward, err := execution.Select(trainingprogram.PhaseForward)
			if err != nil {
				t.Fatal(err)
			}
			example := LinearCTCExample{Hidden: tc.Hidden, Targets: tc.Targets, Frames: tc.Frames}
			if err := gradients.Run(&example); err != nil {
				t.Fatal(err)
			}
			// Reduction-scaled float32 comparison protocol; the independent
			// CPU oracle retains float64 intermediates after float32 input quantization.
			epsilon := float64(math.Nextafter32(1, 2) - 1)
			tolerance := epsilon * float64(tc.Frames+tc.Width+tc.Vocabulary) * max(1, math.Abs(tc.Loss))
			testutil.RequireWithin(t, "projection logits", example.Logits, tc.Logits, tolerance)
			testutil.RequireWithin(t, "projection loss", []float64{example.Loss}, []float64{tc.Loss}, tolerance)
			testutil.RequireElementsWithin(t, "projection gradients", example.Gradient, tc.ProjectionGradient, tolerance)
			testutil.RequireElementsWithin(t, "logits gradients", m.dLogits, tc.LogitsGradient, tolerance)
			gradient := slices.Clone(example.Gradient)
			for index, value := range m.weights {
				const perturbation = float32(1.0 / 256)
				m.weights[index] = value + perturbation
				if err := forward.Run(&example); err != nil {
					t.Fatal(err)
				}
				plus := example.Loss
				m.weights[index] = value - perturbation
				if err := forward.Run(&example); err != nil {
					t.Fatal(err)
				}
				minus := example.Loss
				m.weights[index] = value
				span := 2 * float64(perturbation)
				// Forward float32 rounding is amplified by differencing. The
				// tighter independent autograd comparison above remains mandatory.
				allowance := 2*tolerance/span + span*span
				testutil.RequireWithin(t, "projection finite difference", []float32{gradient[index]}, []float64{(plus - minus) / span}, allowance)
			}
			if err := forward.Run(&example); err != nil {
				t.Fatal(err)
			}
			before := example.Loss
			for step := range tc.Width {
				if err := execution.Run(&example); err != nil || example.Update.Step != step+1 {
					t.Fatalf("shared optimizer step %d: %v", step, err)
				}
			}
			if err := forward.Run(&example); err != nil || example.Loss >= before {
				t.Fatalf("bounded fixture fit = %g -> %g, %v", before, example.Loss, err)
			}
			if !slices.Equal(tc.OutputWeight, frozenWeight) || !slices.Equal(tc.OutputBias, frozenBias) {
				t.Fatal("frozen output projection changed")
			}
			t.Logf("CTC fixture fit %g -> %g in %d shared Muon updates; no held-out quality claim", before, example.Loss, tc.Width)
		})
	}
}

func TestLinearCTCAdmissionAndPhases(t *testing.T) {
	tc := linearCTCOracle(t)[0]
	spec := linearCTCSpec(tc)
	m, err := NewLinearCTC(spec)
	if err != nil {
		t.Fatal(err)
	}
	spec.MemoryBytes = m.StorageBytes()
	if _, err := NewLinearCTC(spec); err != nil {
		t.Fatal(err)
	}
	spec.MemoryBytes--
	if _, err := NewLinearCTC(spec); err == nil {
		t.Fatal("insufficient memory admitted")
	}
	execution, err := m.Bind(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	gradient, err := execution.Select(trainingprogram.PhaseForward, trainingprogram.PhaseBackward)
	if err != nil {
		t.Fatal(err)
	}
	update, err := execution.Select(trainingprogram.PhaseOptimize)
	if err != nil {
		t.Fatal(err)
	}
	example := LinearCTCExample{Hidden: tc.Hidden, Targets: tc.Targets, Frames: tc.Frames}
	if err := update.Run(&example); err == nil {
		t.Fatal("optimizer accepted an absent gradient phase")
	}
	if err := gradient.Run(&example); err != nil {
		t.Fatal(err)
	}
	stale := example
	if err := gradient.Run(&example); err != nil {
		t.Fatal(err)
	}
	if err := update.Run(&stale); err == nil {
		t.Fatal("stale gradient accepted")
	}
	duplicate := example
	if err := update.Run(&example); err != nil {
		t.Fatal(err)
	}
	if err := update.Run(&duplicate); err == nil {
		t.Fatal("gradient consumed twice")
	}
	if err := gradient.Run(&example); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Bind(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := update.Run(&example); err == nil {
		t.Fatal("new binding inherited an incomplete gradient")
	}
	alias := LinearCTCExample{Hidden: m.projected, Targets: tc.Targets, Frames: tc.Frames}
	if err := execution.Run(&alias); err == nil {
		t.Fatal("mutable input alias admitted")
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	bound, err := m.Bind(ctx)
	if err != nil {
		t.Fatal(err)
	}
	weights := m.WeightSnapshot()
	cancel(nil)
	if err := bound.Run(&example); !errors.Is(err, context.Canceled) || !slices.Equal(weights, m.WeightSnapshot()) {
		t.Fatalf("canceled execution changed weights: %v", err)
	}
}

func TestLinearCTCRefusals(t *testing.T) {
	tc := linearCTCOracle(t)[0]
	for name, change := range map[string]func(*LinearCTCSpec){
		"zero-width":     func(s *LinearCTCSpec) { s.Width = 0 },
		"overflow":       func(s *LinearCTCSpec) { s.Width = math.MaxInt },
		"output-shape":   func(s *LinearCTCSpec) { s.OutputWeight = s.OutputWeight[1:] },
		"bias-shape":     func(s *LinearCTCSpec) { s.OutputBias = []float32{0} },
		"negative-blank": func(s *LinearCTCSpec) { s.Blank = -1 },
		"blank-range":    func(s *LinearCTCSpec) { s.Blank = s.Vocabulary },
		"target-bound":   func(s *LinearCTCSpec) { s.MaxTargets = s.MaxFrames + 1 },
		"non-finite-base": func(s *LinearCTCSpec) {
			s.OutputWeight = slices.Clone(s.OutputWeight)
			s.OutputWeight[0] = float32(math.NaN())
		},
		"non-finite-bias": func(s *LinearCTCSpec) {
			s.OutputBias = make([]float32, s.Vocabulary)
			s.OutputBias[0] = float32(math.Inf(1))
		},
		"optimizer": func(s *LinearCTCSpec) { s.Optimizer.BaseLearningRate = math.NaN() },
	} {
		t.Run(name, func(t *testing.T) {
			spec := linearCTCSpec(tc)
			change(&spec)
			if _, err := NewLinearCTC(spec); err == nil {
				t.Fatal("invalid construction admitted")
			}
		})
	}
	spec := linearCTCSpec(tc)
	spec.MaxTargets = spec.MaxFrames
	m, err := NewLinearCTC(spec)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Bind(nil); err == nil {
		t.Fatal("nil context admitted")
	}
	if _, err := (*LinearCTC)(nil).Bind(t.Context()); err == nil {
		t.Fatal("nil component admitted")
	}
	execution, err := m.Bind(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	before := m.WeightSnapshot()
	for name, change := range map[string]func(*LinearCTCExample){
		"empty-frames":    func(e *LinearCTCExample) { e.Frames = 0 },
		"excess-frames":   func(e *LinearCTCExample) { e.Frames = spec.MaxFrames + 1 },
		"input-shape":     func(e *LinearCTCExample) { e.Hidden = e.Hidden[1:] },
		"nan-input":       func(e *LinearCTCExample) { e.Hidden = slices.Clone(e.Hidden); e.Hidden[0] = float32(math.NaN()) },
		"blank-target":    func(e *LinearCTCExample) { e.Targets = []int{spec.Blank} },
		"negative-target": func(e *LinearCTCExample) { e.Targets = []int{-1} },
		"target-range":    func(e *LinearCTCExample) { e.Targets = []int{spec.Vocabulary} },
		"impossible-repeat": func(e *LinearCTCExample) {
			e.Targets = make([]int, e.Frames)
			for i := range e.Targets {
				e.Targets[i] = 1
			}
		},
	} {
		t.Run(name, func(t *testing.T) {
			e := LinearCTCExample{Hidden: tc.Hidden, Targets: tc.Targets, Frames: tc.Frames}
			change(&e)
			if err := execution.Run(&e); err == nil || !slices.Equal(before, m.WeightSnapshot()) {
				t.Fatalf("invalid example reached optimizer: %v", err)
			}
		})
	}
}
