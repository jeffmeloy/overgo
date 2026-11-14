//go:build modeltest

package latentimage

import (
	"context"
	"crypto/sha256"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

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
	generator := testcheck.Must(LoadGenerator(t.Context(), directory, profile, request))(t)
	t.Cleanup(testutil.Closer(t, context.WithoutCancel(t.Context()), generator))
	var library *driver.Library
	testcheck.NoError(t, generator.pipeline.runtime.Do(t.Context(), func(state *device.State) error { library = state.Driver; return nil }))
	session := testcheck.Must(generator.prepare(t.Context(), request))(t)
	if _, err := generator.integrate(t.Context(), session); err != nil {
		t.Fatal(err)
	}
	t.Logf("artifact=%s profile=%s request=%+v", directory, profile.ID, request)
	assertCanceledDecodeStorage(t, generator.pipeline)
	image := testcheck.Must(generator.decode(t.Context(), session))(t)
	if len(image.Data) == 0 {
		t.Fatal("recovered pipeline published no image")
	}
	t.Logf("recovered image SHA256=%x", sha256.Sum256(image.Data))
	testcheck.NoError(t, generator.Close(context.WithoutCancel(t.Context())))
	if stats := library.MemoryStats(); stats.CurrentBytes != 0 || stats.Allocations != 0 {
		t.Fatalf("closed resident pipeline leaked: %+v", stats)
	}
	t.Log("resource: task=Krea-recovery state=not_busy scope=owned-session current_bytes=0")
}
