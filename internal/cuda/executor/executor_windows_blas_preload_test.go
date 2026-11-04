//go:build windows

package executor

import (
	"testing"

	"overgo/internal/cuda/device"
)

// TestAnExecutorPreparesBLASBesideItsUpload holds an executor asked to preload cuBLAS, as a load of a model whose decode
// products take it asks, to its cuBLAS state before any run: the background preload that loads its modules sets cuBLAS
// up too, so the model's first request finds it ready, where Qwen2.5-0.5B's had set it up. An executor not asked opens
// none (TestCellAttentionMatchesReference).
func TestAnExecutorPreparesBLASBesideItsUpload(t *testing.T) {
	// Serial: device fixtures share the one card.
	executor := newFixtureExecutorWithWorker(t, newFixtureWorker(t))
	executor.PreloadBLAS()
	<-executor.preloaded
	if err := executor.worker.Do(t.Context(), func(*device.State) error {
		if executor.resources.blas == nil || executor.resources.blas.library == nil || executor.resources.modules[0] == 0 {
			t.Error("the executor's preload left cuBLAS or its modules for the first request")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}
