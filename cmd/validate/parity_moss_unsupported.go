//go:build !windows

package main

import (
	"flag"
	"fmt"
)

func mossTransformerParity(flags *flag.FlagSet, args []string) error {
	flags.String("model", "", "pinned MOSS-Image-SFT checkpoint directory")
	flags.String("references", "", "native full-reference case directory")
	flags.String("contract", "fixtures/moss", "frozen MOSS checkpoint and oracle contract directory")
	flags.String("stage", "refinement", "native transformer boundary: refinement, joint (first block), or denoiser (complete forward with native conditioning)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	return fmt.Errorf("MOSS refinement verification requires Windows CUDA")
}
