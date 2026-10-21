//go:build windows

package model_test

import (
	"testing"

	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

func TestFlashNextSparseAttentionCUDA(t *testing.T) {
	runFlashNextCUDA(t, model.RunFlashNextSparseAttention)
}

func TestFlashNextNGramCUDA(t *testing.T) {
	runFlashNextCUDA(t, model.RunFlashNextNGram)
}

func TestBlockSelectionChunksCUDA(t *testing.T) {
	runFlashNextCUDA(t, model.RunBlockSelectionChunks)
}

func runFlashNextCUDA(t *testing.T, check func(*testing.T, func([]*tensor.Tensor, map[*tensor.Tensor]reference.Value) (map[*tensor.Tensor]reference.Value, error))) {
	t.Helper()
	cudatest.Require(t)
	device, err := executor.New(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := device.Close(); err != nil {
			t.Error(err)
		}
	})
	check(t, func(outputs []*tensor.Tensor, feeds map[*tensor.Tensor]reference.Value) (map[*tensor.Tensor]reference.Value, error) {
		return device.Execute(t.Context(), outputs, feeds)
	})
}
