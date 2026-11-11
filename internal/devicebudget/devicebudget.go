// Package devicebudget is a declared device-memory budget: a card read as one of the budget's size, so a run on the
// reference card shows what a smaller one would get (card-budgets/declared-budget). It caps memory only: a 16 GiB budget
// on a 48 GiB card tests what fits in 16 GiB, not a 16 GiB card's compute rate, bandwidth or driver.
package devicebudget

import (
	"fmt"
	"os"
	"strconv"
)

// gibibyte is the bytes of one GiB, the unit the budget is declared in; a declaration is read in base 10 (declaredBase)
// and to 32 bits (declaredBits), more GiB than any card holds.
const (
	gibibyte     = 1 << 30
	declaredBase = 10
	declaredBits = 32
)

// Declared is the budget OVERGO_DEVICE_BUDGET_GIB declares in whole GiB, in bytes, zero where none is. The driver reads
// it in place of the card's installed memory, so placement, residency and admission, each of which reads the driver's
// free and total, see the smaller card, and an allocation past it is refused as on that card. It is a result setting of
// every capability's identity (internal/inferencesurface), since placement and chunking under it may change what a
// capability computes, and a run record's environment names it, so a reading under it is kept apart from the full
// card's. The name is spelled where it is read, as a result setting is declared.
func Declared() (uint64, error) {
	text := os.Getenv("OVERGO_DEVICE_BUDGET_GIB")
	if text == "" {
		return 0, nil
	}
	whole, err := strconv.ParseUint(text, declaredBase, declaredBits)
	if err != nil || whole == 0 {
		return 0, fmt.Errorf("device budget: OVERGO_DEVICE_BUDGET_GIB=%q names no whole number of GiB above zero", text)
	}
	return whole * gibibyte, nil
}

// Bound is free and total device memory as a card of budget bytes reports them, where the card holds more: what is in
// use stays in use, out of the budget, and none is free once it holds the budget. A zero budget, or one the card does
// not exceed, leaves both as the card reported them.
func Bound(free, total, budget uint64) (boundFree, boundTotal uint64) {
	if budget == 0 || budget >= total {
		return free, total
	}
	used := total - min(free, total)
	if used >= budget {
		return 0, budget
	}
	return budget - used, budget
}
