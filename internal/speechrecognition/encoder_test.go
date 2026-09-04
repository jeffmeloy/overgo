package speechrecognition

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/binaryschema"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
)

type smallTensor struct {
	Shape  []int     `json:"shape"`
	Values []float32 `json:"values"`
}
type smallCase struct {
	Frames   int                    `json:"frames"`
	Features []float32              `json:"features"`
	Traces   map[string]smallTensor `json:"traces"`
	Logits   []float32              `json:"logits"`
	Tokens   []int                  `json:"tokens"`
}
type smallFixture struct {
	SourceSHA256 string                 `json:"source_sha256"`
	Declaration  Declaration            `json:"declaration"`
	Weights      map[string]smallTensor `json:"weights"`
	Cases        []smallCase            `json:"cases"`
}

func smallEncoder(t *testing.T) (smallFixture, *safetensors.Source, *Encoder) {
	t.Helper()
	encoded, err := os.ReadFile("testdata/encoder.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture smallFixture
	if err = json.Unmarshal(encoded, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) != 3 || len(fixture.SourceSHA256) != 64 {
		t.Fatal("incomplete native oracle")
	}
	values := make(map[string][]float32)
	shapes := make(map[string][]int)
	for name, tensor := range fixture.Weights {
		values[name], shapes[name] = tensor.Values, tensor.Shape
	}
	dir := t.TempDir()
	if err = safetensors.Save(filepath.Join(dir, "model.safetensors"), values, shapes, nil); err != nil {
		t.Fatal(err)
	}
	source, err := safetensors.OpenSource(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = source.Close() })
	e, err := LoadEncoder(t.Context(), source, fixture.Declaration, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	return fixture, source, e
}

func TestEncoderNativeOracle(t *testing.T) {
	fixture, _, e := smallEncoder(t)
	var w Workspace
	for _, tc := range fixture.Cases {
		t.Run(fmt.Sprintf("frames_%d", tc.Frames), func(t *testing.T) {
			seen := 0
			hidden, logits, frames, err := e.Forward(t.Context(), tc.Features, tc.Frames, &w, func(trace Trace) error {
				name := fmt.Sprintf("layer_%d", trace.Block)
				if trace.Block < 0 {
					name = "input_projection"
				}
				want := tc.Traces[name]
				if !slices.Equal(want.Shape, []int{1, trace.Frames, trace.Width}) {
					t.Fatal("trace geometry differs")
				}
				// A float32 dot-product budget for the largest fixture reduction;
				// this protocol is not a global nonlinear-network error bound.
				tolerance := float64(math.Nextafter32(1, 2)-1) * float64(e.blocks[0].first.in.out)
				testutil.RequireWithin(t, name, trace.Values, want.Values, tolerance)
				seen++
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			if seen != len(tc.Traces) || len(hidden) != frames*e.input.out || frames != len(tc.Tokens) {
				t.Fatal("incomplete forward")
			}
			testutil.RequireWithin(t, "logits", logits, tc.Logits, float64(math.Nextafter32(1, 2)-1)*float64(e.blocks[0].first.in.out))
			ids := make([]int, frames)
			if _, err = GreedyCTC(t.Context(), ids, make([]int, frames), logits, frames, e.output.out, 0); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(ids, tc.Tokens) {
				t.Fatalf("tokens=%v want=%v", ids, tc.Tokens)
			}
			before := slices.Clone(logits)
			pointer := &w.x[0]
			_, again, _, err := e.Forward(t.Context(), tc.Features, tc.Frames, &w, nil)
			if err != nil || !slices.Equal(before, again) || pointer != &w.x[0] {
				t.Fatalf("replay differs or reallocates backing storage: %v", err)
			}
		})
	}
}

func TestEncoderRefusesInvalidExecution(t *testing.T) {
	fixture, _, e := smallEncoder(t)
	tc := fixture.Cases[0]
	for _, scenario := range []struct {
		name      string
		ctx       context.Context
		features  []float32
		frames    int
		workspace *Workspace
	}{
		{"nil context", nil, tc.Features, tc.Frames, &Workspace{}},
		{"nil workspace", t.Context(), tc.Features, tc.Frames, nil},
		{"shape", t.Context(), tc.Features[:1], tc.Frames, &Workspace{}},
		{"short pool", t.Context(), tc.Features[:e.input.in], 1, &Workspace{}},
		{"nonfinite", t.Context(), append([]float32{float32(math.NaN())}, tc.Features[1:]...), tc.Frames, &Workspace{}},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			if _, _, _, err := e.Forward(scenario.ctx, scenario.features, scenario.frames, scenario.workspace, nil); err == nil {
				t.Fatal("accepted invalid execution")
			}
		})
	}
	if _, _, _, err := new(Encoder).Forward(t.Context(), tc.Features, tc.Frames, &Workspace{}, nil); err == nil {
		t.Fatal("accepted zero encoder")
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(nil)
	if _, _, _, err := e.Forward(ctx, tc.Features, tc.Frames, &Workspace{}, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation=%v", err)
	}
	sentinel := errors.New("observer stopped")
	if _, _, _, err := e.Forward(t.Context(), tc.Features, tc.Frames, &Workspace{}, func(Trace) error { return sentinel }); !errors.Is(err, sentinel) {
		t.Fatalf("observer error=%v", err)
	}
	var w Workspace
	if _, _, _, err := e.Forward(t.Context(), tc.Features, tc.Frames, &w, nil); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := e.Forward(t.Context(), w.x[:len(tc.Features)], tc.Frames, &w, nil); err == nil {
		t.Fatal("accepted borrowed input")
	}
	other := *e
	if _, _, _, err := other.Forward(t.Context(), tc.Features, tc.Frames, &w, nil); err == nil {
		t.Fatal("accepted another encoder's workspace")
	}
	ctx, cancel = context.WithCancelCause(t.Context())
	if _, _, _, err := e.Forward(ctx, tc.Features, tc.Frames, &w, func(Trace) error { cancel(nil); return nil }); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-forward cancellation=%v", err)
	}
}

func TestEncoderWeightAndWorkspaceBudgets(t *testing.T) {
	fixture, source, e := smallEncoder(t)
	if _, err := LoadEncoder(t.Context(), source, fixture.Declaration, e.weightBytes-1); err == nil {
		t.Fatal("accepted weight budget shortfall")
	}
	exact, err := LoadEncoder(t.Context(), source, fixture.Declaration, e.weightBytes)
	if err != nil {
		t.Fatal(err)
	}
	var w Workspace
	tc := fixture.Cases[0]
	if _, _, _, err = exact.Forward(t.Context(), tc.Features, tc.Frames, &w, nil); err == nil {
		t.Fatal("accepted absent workspace budget")
	}
	for _, buffer := range w.buffers() {
		if cap(*buffer) != 0 {
			t.Fatal("allocated before resource refusal")
		}
	}
	if _, _, _, err = e.Forward(t.Context(), tc.Features, tc.Frames, &w, nil); err != nil {
		t.Fatal(err)
	}
	budget := e.weightBytes
	for _, buffer := range w.buffers() {
		budget += uint64(cap(*buffer)) * binaryschema.Uint32Bytes
	}
	e.memoryBytes = budget
	if _, _, _, err = e.Forward(t.Context(), tc.Features, tc.Frames, &w, nil); err != nil {
		t.Fatal(err)
	}
	e.memoryBytes--
	if _, _, _, err = e.Forward(t.Context(), tc.Features, tc.Frames, &w, nil); err == nil {
		t.Fatal("accepted retained capacity budget shortfall")
	}
}

func TestEncoderDeclarationRefusals(t *testing.T) {
	fixture, source, _ := smallEncoder(t)
	for _, tc := range []struct {
		name   string
		change func(*Declaration)
	}{
		{"epsilon", func(d *Declaration) { d.LayerNormEpsilon = 0 }},
		{"activation", func(d *Declaration) { d.Activation = "unknown" }},
		{"feedback boundary", func(d *Declaration) { d.FeedbackAfter = len(d.Blocks) + 1 }},
		{"feedback disabled with weights", func(d *Declaration) { d.FeedbackAfter = 0 }},
		{"missing weight", func(d *Declaration) { d.Input.Weight = "absent" }},
		{"heads", func(d *Declaration) { d.Blocks[0].Attention.Heads = 3 }},
		{"position table", func(d *Declaration) { d.Blocks[0].Attention.BlockFrames = math.MaxInt }},
		{"stride", func(d *Declaration) { d.Blocks[0].Convolution.Stride = 0 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := fixture.Declaration
			d.Blocks = slices.Clone(d.Blocks)
			tc.change(&d)
			if _, err := LoadEncoder(t.Context(), source, d, 1<<20); err == nil {
				t.Fatal("accepted invalid declaration")
			}
		})
	}
	if _, err := LoadEncoder(nil, source, fixture.Declaration, 1<<20); err == nil {
		t.Fatal("accepted nil context")
	}
	if _, err := LoadEncoder(t.Context(), nil, fixture.Declaration, 1<<20); err == nil {
		t.Fatal("accepted nil source")
	}
}
