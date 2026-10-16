//go:build modeltest && windows

package audioparity

import (
	"testing"

	cudatest "overgo/internal/cuda/testutil"
)

func TestASRResidentReference(t *testing.T) {
	cudatest.Require(t)
	t.Run("connectionist", func(t *testing.T) { testASRReference(t, true) })
	t.Run("recurrent", func(t *testing.T) { testSecondASROffline(t, true) })
	t.Run("streaming", func(t *testing.T) { testSecondASRStreaming(t, true) })
}
