//go:build windows

package latentimage

import (
	"context"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/quant"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// closeReporting closes what a test opened, and reports a close that fails.
func closeReporting(t testing.TB, ctx context.Context, opened interface{ Close(context.Context) error }) {
	if err := opened.Close(ctx); err != nil {
		t.Errorf("close: %v", err)
	}
}

// storedWeight gives a fixture weight's bytes as node stores them: F32 values as they are, others quantized.
func storedWeight(node *tensor.Tensor, values []float32) func() ([]byte, error) {
	return func() ([]byte, error) {
		if node.Type == dtype.F32 {
			return driver.Bytes(values), nil
		}
		return quant.Quantize(node.Type, values)
	}
}
