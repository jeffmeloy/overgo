package inference

import (
	"errors"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/recipe"
)

func TestHybridNativeFallbackRealizedResidency(t *testing.T) {
	oom := &driver.ResultError{Operation: "injected resident allocation", Code: 2}
	partial, releases := false, 0
	recovered, err := loadWithHostRecovery(true, func() error {
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
	actual, err := realizedResidencyForOpen(recipe.ResidencyHybridNative, false, recovered)
	if err != nil || actual != recipe.RealizedOOMStreamed {
		t.Fatalf("fallback outcome = %q, %v", actual, err)
	}
	actual, err = realizedResidencyForOpen(recipe.ResidencyHybridNative, true, false)
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
		got, err := realizedResidencyForOpen(tc.requested, tc.resident, false)
		if err != nil || got != tc.want {
			t.Fatalf("%q realization = %q, %v", tc.requested, got, err)
		}
	}
	if got, err := realizedResidencyForOpen(recipe.ResidencyDeviceNative, false, false); err == nil || got != "" {
		t.Fatalf("missing device preload was accepted: %q, %v", got, err)
	}
	if got := (*Runner)(nil).RealizedResidency(); got != "" {
		t.Fatalf("nil runner claimed residency %q", got)
	}
	if recovered, err := loadWithHostRecovery(false, func() error { return oom }, func() error {
		t.Fatal("non-recoverable policy released for fallback")
		return nil
	}); recovered || !errors.Is(err, oom) {
		t.Fatalf("non-recoverable preload = %v, %v", recovered, err)
	}
	cleanupFailure := errors.New("injected release failure")
	if recovered, err := loadWithHostRecovery(true, func() error { return oom }, func() error { return cleanupFailure }); recovered || !errors.Is(err, oom) || !errors.Is(err, cleanupFailure) {
		t.Fatalf("failed cleanup became streamed success: %v, %v", recovered, err)
	}
}
