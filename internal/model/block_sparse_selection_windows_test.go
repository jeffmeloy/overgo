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
	model.RunFlashNextSparseAttention(t, func(outputs []*tensor.Tensor, feeds map[*tensor.Tensor]reference.Value) (map[*tensor.Tensor]reference.Value, error) {
		return device.Execute(t.Context(), outputs, feeds)
	})
}
