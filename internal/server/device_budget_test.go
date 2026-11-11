package server

import (
	"path/filepath"
	"testing"

	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

// TestADeclaredDeviceBudgetBoundsWhatACardWould holds the server's serving readings to the device budget they run under
// (card-budgets/declared-budget): under a declared 16 GiB budget the environment a serving session records its readings
// in names the budget and is an identity of its own, which the scorecard joins and compares only with itself, so a capped
// reading never stands in a full card's cell; with none declared the session records the environment it always has.
func TestADeclaredDeviceBudgetBoundsWhatACardWould(t *testing.T) {
	const budget = 16 << 30
	store := overgodbtest.Create(t, filepath.Join(t.TempDir(), "repodb"))
	t.Setenv("OVERGO_DEVICE_BUDGET_GIB", "")
	_, full, err := openServingRepository(Config{Repository: store})
	testcheck.NoError(t, err)
	t.Setenv("OVERGO_DEVICE_BUDGET_GIB", "16")
	_, capped, err := openServingRepository(Config{Repository: store})
	testcheck.NoError(t, err)
	if full.DeviceBudgetBytes != 0 {
		t.Errorf("a serving session with no budget declared records %d bytes of one", full.DeviceBudgetBytes)
	}
	if capped.DeviceBudgetBytes != budget {
		t.Errorf("a serving session under a 16 GiB budget records %d bytes; want %d", capped.DeviceBudgetBytes, budget)
	}
	if capped.ID == full.ID {
		t.Errorf("a serving session under a 16 GiB budget records its readings in the full card's environment %s", full.ID)
	}
}
