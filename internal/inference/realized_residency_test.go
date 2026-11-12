package inference

import (
	"errors"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/modeldevice"
	"overgo/internal/recipe"
)

func TestHybridNativeFallbackRealizedResidency(t *testing.T) {
	oom := &driver.ResultError{Operation: "injected resident allocation", Code: 2}
	partial, releases := false, 0
	recovered, err := loadWithHostRecovery(true, false, func() error {
		partial = true
		return oom
	}, func() error {
		if !partial {
			t.Fatal("fallback did not own a partial allocation")
		}
		partial = false
		releases++
		return nil
	})
	if err != nil || !recovered || partial || releases != 1 {
		t.Fatalf("injected OOM fallback = recovered=%v err=%v partial=%v releases=%d", recovered, err, partial, releases)
	}
	actual, err := realizedResidencyForOpen(recipe.ResidencyHybridNative, false, recovered, false)
	if err != nil || actual != recipe.RealizedOOMStreamed {
		t.Fatalf("fallback outcome = %q, %v", actual, err)
	}
	actual, err = realizedResidencyForOpen(recipe.ResidencyHybridNative, true, false, false)
	if err != nil || actual != recipe.RealizedDeviceNative {
		t.Fatalf("successful preload outcome = %q, %v", actual, err)
	}
	for _, tc := range []struct {
		requested recipe.ResidencyPolicy
		resident  bool
		want      recipe.RealizedResidency
	}{
		{recipe.ResidencyStream, false, recipe.RealizedStream},
		{recipe.ResidencyHostCache, false, recipe.RealizedHostCache},
		{recipe.ResidencyHostReference, false, recipe.RealizedHostReference},
		{recipe.ResidencyDeviceF32, true, recipe.RealizedDeviceF32},
		{recipe.ResidencyDeviceNativeBF16, true, recipe.RealizedDeviceNativeBF16},
	} {
		got, err := realizedResidencyForOpen(tc.requested, tc.resident, false, false)
		if err != nil || got != tc.want {
			t.Fatalf("%q realization = %q, %v", tc.requested, got, err)
		}
	}
	if got, err := realizedResidencyForOpen(recipe.ResidencyDeviceNative, false, false, false); err == nil || got != "" {
		t.Fatalf("missing device preload was accepted: %q, %v", got, err)
	}
	if got := (*Runner)(nil).RealizedResidency(); got != "" {
		t.Fatalf("nil runner claimed residency %q", got)
	}
	if got, err := realizedResidencyForOpen(recipe.ResidencyHybridNative, true, false, true); err != nil || got != recipe.RealizedExpertOffloaded {
		t.Fatalf("expert residency = %q, %v", got, err)
	}
	for _, policy := range []recipe.ResidencyPolicy{recipe.ResidencyDeviceNative, recipe.ResidencyStream} {
		if _, err := realizedResidencyForOpen(policy, true, false, true); err == nil {
			t.Fatal("expert residency accepted under another policy")
		}
	}
	if recovered, err := loadWithHostRecovery(false, false, func() error { return oom }, func() error {
		t.Fatal("non-recoverable policy released for fallback")
		return nil
	}); recovered || !errors.Is(err, oom) {
		t.Fatalf("non-recoverable preload = %v, %v", recovered, err)
	}
	cleanupFailure := errors.New("injected release failure")
	if recovered, err := loadWithHostRecovery(true, false, func() error { return oom }, func() error { return cleanupFailure }); recovered || !errors.Is(err, oom) || !errors.Is(err, cleanupFailure) {
		t.Fatalf("failed cleanup became streamed success: %v, %v", recovered, err)
	}
}

// TestARoundsLoadThatDoesNotFitIsRefusedNotStreamed holds a Strata round's load to the card (flash-next-load): a
// round runs on the weights the card holds, so a load the card cannot hold is refused with the out-of-memory error it
// met, and nothing is released to stream; streamed, Qwen3.8 Flash-Next's open on 2026-11-11 left no device weights,
// readied the round anyway, and panicked indexing the layers' cache kinds it had not made.
func TestARoundsLoadThatDoesNotFitIsRefusedNotStreamed(t *testing.T) {
	t.Parallel()
	oom := &driver.ResultError{Operation: "injected expert tier allocation", Code: 2}
	recovered, err := loadWithHostRecovery(true, true, func() error { return oom }, func() error {
		t.Fatal("a round's load released its stores to stream")
		return nil
	})
	if recovered || !errors.Is(err, oom) || !driver.IsOutOfMemory(err) {
		t.Fatalf("a round's load that did not fit gave recovered=%v err=%v; want it refused with its out-of-memory cause", recovered, err)
	}
	other := errors.New("injected file read failure")
	if _, err := loadWithHostRecovery(true, true, func() error { return other }, func() error { return nil }); !errors.Is(err, other) {
		t.Fatalf("a round's load failing otherwise gave %v; want its own error", err)
	}
}

// TestHybridFallbackReleasesPartialStoresWithoutADecodeStore: a model with no
// decode tensors never made a decode store, and the fallback after an
// out-of-memory preload must release the stores it has and recover, not
// dereference the one it lacks.
func TestHybridFallbackReleasesPartialStoresWithoutADecodeStore(t *testing.T) {
	var decode *modeldevice.DeviceConvertedWeights
	raw, f32 := &modeldevice.DeviceWeights{}, &modeldevice.DeviceConvertedWeights{}
	oom := &driver.ResultError{Operation: "injected resident allocation", Code: 2}
	recovered, err := loadWithHostRecovery(true, false, func() error { return oom }, func() error {
		return releasePartialStores(&decode, &raw, &f32)
	})
	if err != nil || !recovered || decode != nil || raw != nil || f32 != nil {
		t.Fatalf("fallback without a decode store = recovered=%v err=%v stores=%v/%v/%v", recovered, err, decode, raw, f32)
	}
}
