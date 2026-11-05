package scratchtrain

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/adaptiveparity"
	"overgo/internal/testcheck"
)

// loadOracle reads the adaptive scratch oracle the host and resident
// trainers are compared on.
func loadOracle(t *testing.T) adaptiveparity.ScratchOracle {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "fixtures", "adaptive_scratch_oracle.json"))
	testcheck.NoError(t, err)
	oracle, _, err := adaptiveparity.NormalizeScratchOracle(raw)
	testcheck.NoError(t, err)
	return oracle
}
