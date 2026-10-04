package scratchtrain

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/adaptiveparity"
)

// loadOracle reads the adaptive scratch oracle the host and resident
// trainers are compared on.
func loadOracle(t *testing.T) adaptiveparity.ScratchOracle {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "fixtures", "adaptive_scratch_oracle.json"))
	if err != nil {
		t.Fatal(err)
	}
	oracle, _, err := adaptiveparity.NormalizeScratchOracle(raw)
	if err != nil {
		t.Fatal(err)
	}
	return oracle
}
