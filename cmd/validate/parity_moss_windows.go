//go:build windows

package main

import (
	"context"
	"flag"
	"fmt"

	"overgo/internal/latentimage"
)

func mossRefinementParity(flags *flag.FlagSet, args []string) error {
	model := flags.String("model", "", "pinned MOSS-Image-SFT checkpoint directory")
	references := flags.String("references", "", "native full-reference case directory")
	contract := flags.String("contract", "fixtures/moss", "frozen MOSS checkpoint and oracle contract directory")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *model == "" || *references == "" || flags.NArg() != 0 {
		return fmt.Errorf("MOSS refinement requires -model and -references with no positional arguments")
	}
	return latentimage.VerifyMOSSRefinement(context.Background(), *model, *references, *contract, flags.Output())
}
