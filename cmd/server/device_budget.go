package main

import (
	"fmt"

	"overgo/internal/cuda/driver"
	"overgo/internal/devicebudget"
	"overgo/internal/runrecord"
)

// budgetedEnvironment is the environment the server's serving readings are recorded under where a device budget is
// declared: the card they run on, by name and installed memory, beside the budget the environment names
// (card-budgets/declared-budget), so a capped reading says what it was capped to and on what card, a cap on the
// reference card and not a smaller card. With no budget it is none, and the server records the process's environment
// as before.
func budgetedEnvironment(ordinal int) (runrecord.Environment, error) {
	budget, err := devicebudget.Declared()
	if err != nil || budget == 0 {
		return runrecord.Environment{}, err
	}
	library, err := driver.Open()
	if err != nil {
		return runrecord.Environment{}, err
	}
	if err := library.Init(); err != nil {
		return runrecord.Environment{}, err
	}
	card, err := library.DeviceInfo(ordinal)
	if err != nil {
		return runrecord.Environment{}, err
	}
	return runrecord.CurrentEnvironment(fmt.Sprintf("cuda:%d %s, %d bytes", ordinal, card.Name, card.TotalMemoryBytes), "cuda")
}
