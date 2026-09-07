package speechrecognition

import (
	"context"
	"errors"
	"slices"
	"testing"

	"overgo/internal/adaptertrain"
	"overgo/internal/binaryschema"
	"overgo/internal/trainingprogram"
)

func TestEncoderOutputAdapter(t *testing.T) {
	fixture, _, e := smallEncoder(t)
	maxFrames := 0
	for _, tc := range fixture.Cases {
		maxFrames = max(maxFrames, tc.Frames)
	}
	var w Workspace
	adapter, err := e.NewOutputAdapter(t.Context(), &w, maxFrames, 1, 0, trainingprogram.BuiltinOptimizerPolicy())
	if err != nil {
		t.Fatal(err)
	}
	execution, err := adapter.Bind(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	forward, err := execution.Select(trainingprogram.PhaseForward)
	if err != nil {
		t.Fatal(err)
	}
	frozenWeight, frozenBias := slices.Clone(e.output.weight), slices.Clone(e.output.bias)
	for _, tc := range fixture.Cases {
		hidden, frames, err := e.Encode(t.Context(), tc.Features, tc.Frames, &w, nil)
		if err != nil {
			t.Fatal(err)
		}
		base, err := e.Project(t.Context(), hidden, frames, &w)
		if err != nil {
			t.Fatal(err)
		}
		example := adaptertrain.LinearCTCExample{Hidden: hidden, Targets: []int{1}, Frames: frames}
		if err := forward.Run(&example); err != nil || !slices.Equal(base, example.Logits) {
			t.Fatalf("identity projection changed encoder logits: %v", err)
		}
	}
	tc := fixture.Cases[0]
	hidden, frames, err := e.Encode(t.Context(), tc.Features, tc.Frames, &w, nil)
	if err != nil {
		t.Fatal(err)
	}
	base, err := e.Project(t.Context(), hidden, frames, &w)
	if err != nil {
		t.Fatal(err)
	}
	beforeHidden, beforeBase := slices.Clone(hidden), slices.Clone(base)
	beforeAdapter := adapter.WeightSnapshot()
	example := adaptertrain.LinearCTCExample{Hidden: hidden, Targets: []int{1}, Frames: frames}
	if err := execution.Run(&example); err != nil || example.Update.Step != 1 {
		t.Fatalf("compiled CTC update: %v", err)
	}
	if slices.Equal(beforeAdapter, adapter.WeightSnapshot()) {
		t.Fatal("input projection did not train")
	}
	if !slices.Equal(frozenWeight, e.output.weight) || !slices.Equal(frozenBias, e.output.bias) {
		t.Fatal("shared final/intermediate output projection was modified")
	}
	hidden, frames, err = e.Encode(t.Context(), tc.Features, tc.Frames, &w, nil)
	if err != nil {
		t.Fatal(err)
	}
	base, err = e.Project(t.Context(), hidden, frames, &w)
	if err != nil || !slices.Equal(beforeHidden, hidden) || !slices.Equal(beforeBase, base) {
		t.Fatalf("frozen encoder or tied feedback changed: %v", err)
	}
	if err := forward.Run(&example); err != nil || slices.Equal(base, example.Logits) {
		t.Fatalf("trained adapter has no output effect: %v", err)
	}
	admitted := e.weightBytes + adapter.StorageBytes()
	for _, buffer := range w.buffers() {
		admitted += uint64(cap(*buffer)) * binaryschema.Uint32Bytes
	}
	e.memoryBytes = admitted
	if _, _, err := e.Encode(t.Context(), tc.Features, tc.Frames, &w, nil); err != nil {
		t.Fatal(err)
	}
	e.memoryBytes--
	if _, _, err := e.Encode(t.Context(), tc.Features, tc.Frames, &w, nil); err == nil {
		t.Fatal("encoder ignored the adapter's optimizer/loss reservation")
	}
	if _, err := e.NewOutputAdapter(t.Context(), &w, maxFrames, 1, 0, trainingprogram.BuiltinOptimizerPolicy()); err == nil {
		t.Fatal("workspace rebound while its adapter remains live")
	}
}

func TestOutputProjectionAdmission(t *testing.T) {
	fixture, _, e := smallEncoder(t)
	tc := fixture.Cases[0]
	var w Workspace
	hidden, frames, err := e.Encode(t.Context(), tc.Features, tc.Frames, &w, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name string
		call func() error
	}{
		{"nil-context", func() error { _, err := e.Project(nil, hidden, frames, &w); return err }},
		{"nil-workspace", func() error { _, err := e.Project(t.Context(), hidden, frames, nil); return err }},
		{"unbound-workspace", func() error { _, err := e.Project(t.Context(), hidden, frames, new(Workspace)); return err }},
		{"shape", func() error { _, err := e.Project(t.Context(), hidden[:len(hidden)-1], frames, &w); return err }},
		{"empty", func() error { _, err := e.Project(t.Context(), nil, 0, &w); return err }},
		{"overlap", func() error { _, err := e.Project(t.Context(), w.logits[:e.output.in], 1, &w); return err }},
	} {
		t.Run(test.name, func(t *testing.T) {
			if test.call() == nil {
				t.Fatal("malformed projection admitted")
			}
		})
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(nil)
	if _, err := e.Project(ctx, hidden, frames, &w); !errors.Is(err, context.Canceled) {
		t.Fatalf("projection cancellation = %v", err)
	}
	var unbound Workspace
	if _, err := e.NewOutputAdapter(ctx, &unbound, tc.Frames, 1, 0, trainingprogram.BuiltinOptimizerPolicy()); !errors.Is(err, context.Canceled) || unbound.owner != nil {
		t.Fatalf("adapter cancellation = %v", err)
	}
	e.memoryBytes = e.weightBytes
	if _, err := e.NewOutputAdapter(t.Context(), &unbound, tc.Frames, 1, 0, trainingprogram.BuiltinOptimizerPolicy()); err == nil || unbound.adapter != nil {
		t.Fatal("unfunded adapter admitted")
	}
	for _, buffer := range unbound.buffers() {
		if cap(*buffer) != 0 {
			t.Fatal("unfunded adapter allocated encoder buffers")
		}
	}
}
