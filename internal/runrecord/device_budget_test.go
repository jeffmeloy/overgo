package runrecord

import (
	"testing"

	"overgo/internal/testcheck"
)

// TestADeclaredDeviceBudgetBoundsWhatACardWould holds a run's environment to the device budget it ran under
// (card-budgets/declared-budget): under a 16 GiB budget the environment names it and is an identity of its own, so a
// scorecard, which joins and compares readings only of one environment, keeps a capped reading apart from the full
// card's; with none declared the environment names none and keeps the identity it had before budgets were.
func TestADeclaredDeviceBudgetBoundsWhatACardWould(t *testing.T) {
	const budget = 16 << 30
	t.Setenv("OVERGO_DEVICE_BUDGET_GIB", "")
	full := testcheck.Must(CurrentEnvironment("process", "go"))(t)
	if full.DeviceBudgetBytes != 0 {
		t.Errorf("an environment with no budget declared names %d bytes of one", full.DeviceBudgetBytes)
	}
	before := testcheck.Must(NewEnvironment(Environment{Host: full.Host, OS: full.OS, Arch: full.Arch, Device: full.Device,
		Backend: full.Backend, Driver: full.Driver, Runtime: full.Runtime}))(t)
	if full.ID != before.ID {
		t.Errorf("an environment with no budget is %s, not the identity %s it had before budgets were", full.ID, before.ID)
	}
	t.Setenv("OVERGO_DEVICE_BUDGET_GIB", "16")
	capped := testcheck.Must(CurrentEnvironment("process", "go"))(t)
	if capped.DeviceBudgetBytes != budget {
		t.Errorf("an environment under a 16 GiB budget names %d bytes; want %d", capped.DeviceBudgetBytes, budget)
	}
	if capped.ID == full.ID {
		t.Errorf("an environment under a 16 GiB budget is %s, the full card's own identity", capped.ID)
	}
}
