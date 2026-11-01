//go:build windows

package latentimage

import (
	cudatest "overgo/internal/cuda/testutil"
	"testing"
)

func TestMOSSConditioningDeltaGolden(t *testing.T) {
	t.Run("cpu", TestMOSSConditioningDeltaCPU)
	cudatest.Require(t)
	run, _, close := cudaVAERunner(t)
	defer close()
	t.Run("cuda", func(t *testing.T) {
		conditioningDeltaGolden(t, "chunk-state-cuda.safetensors", "47debabb50798649d105eeef9056285396edc3a1602323a6dc36098c51b19773", run)
	})
}
