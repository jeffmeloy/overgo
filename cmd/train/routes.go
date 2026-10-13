package main

import (
	"context"
	"flag"
	"fmt"
	"maps"
	"slices"
	"strings"
)

// routeFlag selects a bounded family trainer by name, ahead of every other
// flag: train -route <name> [its flags]. Each route parses its own flags.
const routeFlag = "-route"

// proofFlag registers -proof for the dense workflow, train -objective and
// each route the measurement run proves that had no -proof of its own.
func proofFlag(flags *flag.FlagSet) *bool {
	return flags.Bool("proof", false, "re-verify the recipe by the training run alone: the store is given the trained checkpoint's identity, no file is kept at -out, and no held-out verdict publishes")
}

// trainRoutes are the bounded family trainers that were once their own
// *-train-probe commands; the trainer is reached through cmd/train by name
// and trains until its context ends. A route that needs a platform's device
// registers itself from that platform's file.
var trainRoutes = map[string]func(ctx context.Context, flags *flag.FlagSet, args []string) error{
	"diffusionimage":  trainDiffusionImage,
	"codec":           trainCodec,
	"dit":             trainDiT,
	"encoder-bridge":  trainEncoderBridge,
	"flow-organ":      trainFlowOrgan,
	"graft":           trainGraft,
	"mixture":         trainMixture,
	"mot":             trainMot,
	"oscillatorimage": trainOscillatorImage,
	"seq2seq":         trainSeq2Seq,
	"seriesforecast":  trainSeriesForecast,
	"speech":          trainSpeech,
	"tabular":         trainTabular,
	"thoughtbank":     trainThoughtBank,
}

// runRoute dispatches args (the route name, then its flags) to the named
// trainer under ctx, parsing its flags on a set named for the route.
func runRoute(ctx context.Context, args []string) error {
	names := slices.Sorted(maps.Keys(trainRoutes))
	if len(args) == 0 {
		return fmt.Errorf("%s needs a trainer name: %s", routeFlag, strings.Join(names, ", "))
	}
	route, known := trainRoutes[args[0]]
	if !known {
		return fmt.Errorf("%s %q is not a trainer; routes: %s", routeFlag, args[0], strings.Join(names, ", "))
	}
	return route(ctx, flag.NewFlagSet(routeFlag+" "+args[0], flag.ContinueOnError), args[1:])
}
