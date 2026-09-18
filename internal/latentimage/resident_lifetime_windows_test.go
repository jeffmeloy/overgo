package latentimage

import (
	"context"
	"crypto/sha256"
	"errors"
	"slices"
	"sync/atomic"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/graphruntime"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testutil"
)

type lifetimeContext struct {
	context.Context
	onCheck func()
}

func (ctx lifetimeContext) Err() error {
	ctx.onCheck()
	return ctx.Context.Err()
}

func TestResidentFixtureCleanupUsesLiveContext(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	input := builder.Input("cleanup", dtype.F32, tensor.MustShape(1))
	fixture := newResidentFixture(t, t.Context(), "cleanup fixture", t.TempDir(), nil, builder.Scale(input, 2))
	if err := fixture.runtime.BindF32(t.Context(), fixture.graph, "cleanup fixture", input, []float32{1}); err != nil {
		t.Fatal(err)
	}
	// testing cancels t.Context before invoking the helper's cleanup callback.
	// The bound weight requires worker submission even after that cancellation.
}

func TestResidentDecodeCancellationReusesStorage(t *testing.T) {
	cudatest.Require(t)
	for _, name := range []string{"first-session", "second-session"} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			checkDecodeLifetimeFixture(t)
		})
	}
}

func checkDecodeLifetimeFixture(t *testing.T) {
	t.Helper()
	runtime, err := graphruntime.NewResidentSession(0)
	if err != nil {
		t.Fatal(err)
	}
	pipeline := &ResidentImagePipeline{runtime: runtime}
	t.Cleanup(func() {
		if err := pipeline.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Error(err)
		}
	})
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(4)
	input := builder.Input("input", dtype.F32, shape)
	upload := builder.Scale(input, 1)
	bridge := builder.Scale(input, 1)
	decoded := builder.Scale(input, 2)
	compile := func(dynamic []*tensor.Tensor, output *tensor.Tensor) *graphruntime.ResidentProgram {
		t.Helper()
		program, err := runtime.Compile(t.Context(), "lifetime fixture", dynamic, output)
		if err != nil {
			t.Fatal(err)
		}
		return program
	}
	uploader := compile(nil, upload)
	pipeline.vaeBridge, pipeline.vae = compile([]*tensor.Tensor{input}, bridge), compile([]*tensor.Tensor{input}, decoded)
	pipeline.vaeOutput = bridge
	pipeline.VAE = &VAEProgram{Output: decoded, OutH: 1, OutW: 4}
	initial, err := runtime.Retain(t.Context(), uploader, map[*tensor.Tensor]reference.Value{
		input: {Shape: shape, Data: []float32{1, -2, 0, 0.125}},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	value, ok := initial.Value(upload)
	if !ok {
		t.Fatal("fixture upload is unavailable")
	}
	pipeline.latent = &residentLatent{outputs: initial, node: upload, value: value}
	assertCanceledDecodeStorage(t, pipeline)
}

func TestResidentLifetimeResourceAcceptance(t *testing.T) {
	cudatest.Require(t)
	directory := testutil.ModelArtifactDir(t, "Krea-2-Turbo")
	profile, err := ResolveProfile(directory)
	if err != nil {
		t.Fatalf("required resident pipeline artifact: %v", err)
	}
	// Exact output and owned allocation checks permit shared GPU admission.
	// Use the existing bounded real-pipeline cohort, including all denoise
	// steps, before exercising decoder cancellation and subsequent publication.
	request := Request{
		Prompt: kreaGoldenPrompt,
		Width:  256, Height: 256, Steps: 8, Seed: 42, DynamicShiftMu: 1.15,
	}
	generator, err := LoadGenerator(t.Context(), directory, profile, request)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := generator.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Error(err)
		}
	})
	var library *driver.Library
	if err := generator.pipeline.runtime.Do(t.Context(), func(state *device.State) error { library = state.Driver; return nil }); err != nil {
		t.Fatal(err)
	}
	session, err := generator.prepare(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := generator.integrate(t.Context(), session); err != nil {
		t.Fatal(err)
	}
	t.Logf("artifact=%s profile=%s request=%+v", directory, profile.ID, request)
	assertCanceledDecodeStorage(t, generator.pipeline)
	image, err := generator.decode(t.Context(), session)
	if err != nil {
		t.Fatal(err)
	}
	if len(image.Data) == 0 {
		t.Fatal("recovered pipeline published no image")
	}
	t.Logf("recovered image SHA256=%x", sha256.Sum256(image.Data))
	if err := generator.Close(context.WithoutCancel(t.Context())); err != nil {
		t.Fatal(err)
	}
	if stats := library.MemoryStats(); stats.CurrentBytes != 0 || stats.Allocations != 0 {
		t.Fatalf("closed resident pipeline leaked: %+v", stats)
	}
	t.Log("resource: task=Krea-recovery state=not_busy scope=owned-session current_bytes=0")
}

func assertCanceledDecodeStorage(t *testing.T, pipeline *ResidentImagePipeline) {
	t.Helper()
	want, height, width, err := pipeline.DecodeHWC(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("decoder output SHA256=%x extent=%dx%d", sha256.Sum256(driver.Bytes(want)), width, height)
	// Learn the bridge's context-check count on its warm execution path. The
	// next check belongs to decoder submission, after the bridge owns a lease.
	var bridgeChecks atomic.Int64
	control := lifetimeContext{Context: t.Context(), onCheck: func() { bridgeChecks.Add(1) }}
	probe, err := pipeline.runtime.Retain(control, pipeline.vaeBridge, nil, []driver.DevicePtr{pipeline.latent.value.Pointer})
	if err != nil {
		t.Fatal(err)
	}
	if err := probe.Release(t.Context()); err != nil {
		t.Fatal(err)
	}
	if bridgeChecks.Load() == 0 {
		t.Fatal("bridge did not exercise cancellation checks")
	}
	before, err := pipeline.runtime.Stats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	for range 3 {
		ctx, cancel := context.WithCancelCause(t.Context())
		var checks atomic.Int64
		interrupted := lifetimeContext{Context: ctx, onCheck: func() {
			if checks.Add(1) > bridgeChecks.Load() {
				cancel(context.Canceled)
			}
		}}
		_, _, _, err := pipeline.DecodeHWC(interrupted)
		cancel(nil)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("decode interruption: %v", err)
		}
	}
	got, gotHeight, gotWidth, err := pipeline.DecodeHWC(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if height != gotHeight || width != gotWidth || !slices.Equal(got, want) {
		t.Fatal("canceled decode changed subsequent output")
	}
	after, err := pipeline.runtime.Stats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("decode cancellation: bridge_checks=%d owned_bytes=%d->%d allocations=%d->%d owned_peak=%d->%d",
		bridgeChecks.Load(), before.Device.CurrentBytes, after.Device.CurrentBytes,
		before.Device.Allocations, after.Device.Allocations, before.Device.PeakBytes, after.Device.PeakBytes)
	if after.Device.CurrentBytes != before.Device.CurrentBytes || after.Device.Allocations != before.Device.Allocations {
		t.Fatal("canceled decode lost reusable storage")
	}
}
