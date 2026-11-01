//go:build windows

package executor

import (
	"context"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestSpatialConvolutionResidency(t *testing.T) {
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	for _, width := range []uint64{7, 13, 7} {
		b := tensor.NewBuilder()
		x := b.Input("x", dtype.F32, tensor.MustShape(8, width, 5))
		w := b.Input("w", dtype.F32, tensor.MustShape(3, 3, 8, 8))
		out := b.Conv2DWinogradF32(x, w, nil)
		compiled, err := Compile(out)
		if err != nil {
			t.Fatal(err)
		}
		if compiled.matmulStagingBytes != 0 {
			t.Fatal("Winograd retained unused im2col staging")
		}
		if err = cuda.CheckRetainedMemory(t.Context(), compiled, nil, nil, nil); err != nil {
			t.Fatal(err)
		}
		feeds := map[*tensor.Tensor]reference.Value{x: patternedValue(x.Shape, 1, 0.1, 0), w: patternedValue(w.Shape, 2, 0.1, 0)}
		if _, err = cuda.Execute(t.Context(), []*tensor.Tensor{out}, feeds); err != nil {
			t.Fatal(err)
		}
		if err = cuda.worker.Do(t.Context(), func(state *device.State) error {
			plans := cuda.resources.blas.convolutionPlans
			if len(plans) != 1 {
				t.Errorf("retained %d convolution plans, want current shape only", len(plans))
			}
			for shape := range plans {
				if shape.Width != int(width) {
					t.Error("obsolete convolution plan retained")
				}
			}
			return nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	if err := cuda.Close(); err != nil {
		t.Fatal(err)
	}
	stats, err := worker.MemoryStats(context.WithoutCancel(t.Context()))
	if err != nil {
		t.Fatal(err)
	}
	if stats.CurrentBytes != 0 {
		t.Fatalf("executor close retained %d device bytes", stats.CurrentBytes)
	}
}

func TestMemoryEfficientAttentionAvoidsScoreStaging(t *testing.T) {
	b := tensor.NewBuilder()
	x := b.Input("x", dtype.F32, tensor.MustShape(384, 1, 257))
	out := b.AttentionWithOptions(x, x, x, tensor.AttentionOptions{Scale: 1, MemoryEfficientF32: true})
	compiled, err := Compile(out)
	if err != nil {
		t.Fatal(err)
	}
	if compiled.attentionScoreBytes != 0 {
		t.Fatalf("retained %d unused score bytes", compiled.attentionScoreBytes)
	}
}
