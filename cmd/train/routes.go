package main

import (
	"context"
	"flag"
	"fmt"
	"maps"
	"os"
	"slices"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/runrecord"
	"overgo/internal/trainingworkflow"
)

// routeFlag selects a bounded family trainer by name, ahead of every other
// flag: train -route <name> [its flags]. Each route parses its own flags.
const routeFlag = "-route"

// proofFlag registers -proof for the dense workflow, train -objective and
// each route the measurement run proves that had no -proof of its own.
func proofFlag(flags *flag.FlagSet) *bool {
	return flags.Bool("proof", false, "re-verify the recipe by the training run alone: the store is given the trained checkpoint's identity, no file is kept at -out, and no held-out verdict publishes")
}

// codeRevision names the code revision a run executes, as the verification of a proof binds it
// (trainingworkflow.Observer.Verify reads runrecord.ExecutableCodeCommit).
var codeRevision = runrecord.ExecutableCodeCommit

// parseProving parses a route's flags and refuses a -proof run that could not verify its recipe (provable).
func parseProving(flags *flag.FlagSet, args []string, proof *bool) error {
	if err := flags.Parse(args); err != nil {
		return err
	}
	return provable(*proof)
}

// provable refuses a -proof run whose verification could not bind its code revision where the validator awaits it, its
// preflight or its acquisition, before the admission says it is ready or the card is used: a -proof run from a modified
// tree had prepared and trained on the card, then printed that it verified nothing and exited 0 (SQA-522). A -proof run
// by hand is told so before it starts and trains as before; training without -proof is not asked.
func provable(proof bool) error {
	if !proof {
		return nil
	}
	_, err := codeRevision(".")
	switch {
	case err == nil:
		return nil
	case os.Getenv(clioptions.PreflightEnvironment) != "" || os.Getenv(trainingworkflow.VerificationOutputEnvironment) != "":
		return fmt.Errorf("-proof verifies its recipe under the code revision the run executes, and this run can name none: %w", err)
	}
	fmt.Printf("train: this -proof run will verify no recipe, since it can name no code revision: %v\n", err)
	return nil
}

// trainRoutes are the bounded family trainers that were once their own
// *-train-probe commands; the trainer is reached through cmd/train by name
// and trains until its context ends. A route that needs a platform's device
// registers itself from that platform's file.
var trainRoutes = map[string]func(ctx context.Context, flags *flag.FlagSet, args []string) error{
	"diffusionimage":    trainDiffusionImage,
	"caption-projector": trainCaptionProjector,
	"codec":             trainCodec,
	"dit":               trainDiT,
	"encoder-bridge":    trainEncoderBridge,
	"flow-organ":        trainFlowOrgan,
	"graft":             trainGraft,
	"mixture":           trainMixture,
	"mot":               trainMot,
	"oscillatorimage":   trainOscillatorImage,
	"seq2seq":           trainSeq2Seq,
	"seriesforecast":    trainSeriesForecast,
	"speech":            trainSpeech,
	"tabular":           trainTabular,
	"thoughtbank":       trainThoughtBank,
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
