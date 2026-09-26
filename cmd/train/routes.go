package main

import (
	"flag"
	"fmt"
	"maps"
	"slices"
	"strings"
)

// routeFlag selects a bounded family trainer by name, ahead of every other
// flag: train -route <name> [its flags]. Each route parses its own flags.
const routeFlag = "-route"

// trainRoutes are the bounded family trainers that were once their own
// *-train-probe commands; the trainer is reached through cmd/train by name.
var trainRoutes = map[string]func(flags *flag.FlagSet, args []string) error{
	"oscillatorimage": trainOscillatorImage,
	"tabular":         trainTabular,
}

// runRoute dispatches args (the route name, then its flags) to the named
// trainer, parsing its flags on a set named for the route.
func runRoute(args []string) error {
	names := slices.Sorted(maps.Keys(trainRoutes))
	if len(args) == 0 {
		return fmt.Errorf("%s needs a trainer name: %s", routeFlag, strings.Join(names, ", "))
	}
	route, known := trainRoutes[args[0]]
	if !known {
		return fmt.Errorf("%s %q is not a trainer; routes: %s", routeFlag, args[0], strings.Join(names, ", "))
	}
	return route(flag.NewFlagSet(routeFlag+" "+args[0], flag.ContinueOnError), args[1:])
}
