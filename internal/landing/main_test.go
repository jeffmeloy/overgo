package landing

import (
	"os"
	"testing"
)

// TestMain names a test turn of the package's own for its fixture landings (testTurn), so these tests run by hand beside
// a run of the cycle's lanes do not wait out its packages; a test that holds the turn to a process names its own.
func TestMain(m *testing.M) {
	testTurn = func() string { return "overgo-landing-fixtures-test-turn" }
	os.Exit(m.Run())
}
