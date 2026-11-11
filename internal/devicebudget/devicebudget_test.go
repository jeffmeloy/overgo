package devicebudget

import (
	"testing"
)

// TestADeclaredDeviceBudgetBoundsWhatACardWould holds the budget to the card it stands for (card-budgets/
// declared-budget): a 48 GiB card with 10 GiB in use, read under 16 GiB, reports 16 in all and 6 free; under 8 it
// reports none free, its use past the budget; and no budget, or one the card does not exceed, leaves the card as it
// reported itself. A declaration is a whole number of GiB above zero, and anything else is refused, not read as none.
func TestADeclaredDeviceBudgetBoundsWhatACardWould(t *testing.T) {
	const card, used = 48 * gibibyte, 10 * gibibyte
	for _, bound := range []struct {
		budget, free, total uint64
	}{
		{budget: 16 * gibibyte, free: 6 * gibibyte, total: 16 * gibibyte},
		{budget: 8 * gibibyte, free: 0, total: 8 * gibibyte},
		{budget: 0, free: card - used, total: card},
		{budget: 64 * gibibyte, free: card - used, total: card},
	} {
		free, total := Bound(card-used, card, bound.budget)
		if free != bound.free || total != bound.total {
			t.Errorf("a %d-byte card with %d in use under a %d-byte budget reads %d free of %d; want %d of %d",
				uint64(card), uint64(used), bound.budget, free, total, bound.free, bound.total)
		}
	}
	for text, want := range map[string]uint64{"": 0, "16": 16 * gibibyte, "24": 24 * gibibyte} {
		t.Setenv("OVERGO_DEVICE_BUDGET_GIB", text)
		if got, err := Declared(); err != nil || got != want {
			t.Errorf("OVERGO_DEVICE_BUDGET_GIB=%q declares %d (%v); want %d", text, got, err, want)
		}
	}
	for _, text := range []string{"0", "-16", "16GiB", "1.5"} {
		t.Setenv("OVERGO_DEVICE_BUDGET_GIB", text)
		if got, err := Declared(); err == nil {
			t.Errorf("OVERGO_DEVICE_BUDGET_GIB=%q declared %d; want it refused", text, got)
		}
	}
}
