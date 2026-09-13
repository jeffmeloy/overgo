package diffusionimage

import (
	"context"
	"errors"
	"slices"
	"sync/atomic"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// samplingCancellation stops at a check count learned from a completed
// single-step control. Host and device sampling use the same test mechanism.
type samplingCancellation struct {
	context.Context
	checks atomic.Int64
	limit  int64
	cancel context.CancelCauseFunc
}

func (ctx *samplingCancellation) Err() error {
	if count := ctx.checks.Add(1); ctx.limit > 0 && count >= ctx.limit {
		ctx.cancel(context.Canceled)
	}
	return ctx.Context.Err()
}

func TestHostSamplingCancellationRecovery(t *testing.T) {
	fixture := loadGolden[uditTinyFixture](t, "udit_tiny")
	model := compileTiny(t, fixture)
	// Two iterations provide a boundary after one completed forward pass.
	const steps, seed = 2, int64(7)
	want, err := model.Sample(t.Context(), fixture.B, fixture.H, fixture.W, steps, seed)
	if err != nil {
		t.Fatal(err)
	}
	probe := &samplingCancellation{Context: t.Context()}
	if _, err := model.Sample(probe, fixture.B, fixture.H, fixture.W, 1, seed); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	defer cancel(nil)
	boundary := &samplingCancellation{Context: ctx, limit: probe.checks.Load() + 1, cancel: cancel}
	if _, err := model.Sample(boundary, fixture.B, fixture.H, fixture.W, steps, seed); !errors.Is(err, context.Canceled) {
		t.Fatal("host sampler ignored work-boundary cancellation", err)
	}
	got, err := model.Sample(t.Context(), fixture.B, fixture.H, fixture.W, steps, seed)
	if err != nil || !slices.Equal(got, want) {
		t.Fatal("cancellation changed subsequent host sample", err)
	}
}

func TestResidentGeneratorCancellationBeforeLoad(t *testing.T) {
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	request := Request{Height: 1, Width: 1, Steps: 1}
	if _, err := LoadResidentGenerator(ctx, t.TempDir(), request); !errors.Is(err, context.Canceled) {
		t.Fatal("canceled request reached artifact loading", err)
	}
	var generator *ResidentGenerator
	if err := generator.Reset(ctx, request); !errors.Is(err, context.Canceled) {
		t.Fatal("reset discarded request cancellation", err)
	}
}

func TestResidentSamplingContractAcceptance(t *testing.T) {
	// The scalar is broadcast without changing the reference's two rounded
	// operations. An explicit F32 conversion prevents a host fused multiply-add.
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(3, 2)
	input := builder.Input("input", dtype.F32, shape)
	velocity := builder.Input("velocity", dtype.F32, shape)
	delta, next := samplingUpdate(builder, input, velocity)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	x := []float32{1, -2, 0, 7, -8, 0.125}
	v := []float32{-3, 2, 9, -0.125, 4, 11}
	for _, steps := range []int{1, 3, 8} {
		dt := float32(1) / float32(steps)
		result, err := reference.Execute([]*tensor.Tensor{next}, map[*tensor.Tensor]reference.Value{
			input: {Shape: shape, Data: x}, velocity: {Shape: shape, Data: v},
			delta: {Shape: delta.Shape, Data: []float32{dt}},
		})
		if err != nil {
			t.Fatal(err)
		}
		for index, value := range result[next].Data {
			if want := x[index] + float32(v[index]*dt); value != want {
				t.Fatalf("steps=%d element=%d got=%g want=%g", steps, index, value, want)
			}
		}
	}
	var unavailable *ResidentForward
	if _, err := unavailable.Sample(t.Context(), 1, 0); err == nil {
		t.Fatal("unavailable sampler accepted")
	}
	if err := unavailable.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
}
