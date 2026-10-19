//go:build windows

package main

import (
	"context"
	"flag"
	"fmt"

	"overgo/internal/latentimage"
)

func mossTransformerParity(flags *flag.FlagSet, args []string) error {
	model := flags.String("model", "", "pinned MOSS-Image-SFT checkpoint directory")
	references := flags.String("references", "", "native full-reference case directory")
	contract := flags.String("contract", "fixtures/moss", "frozen MOSS checkpoint and oracle contract directory")
	stage := flags.String("stage", "refinement", "native transformer boundary: refinement, joint (first block), denoiser (complete forward), sampling (complete trajectory), decoder (pixels), conditioning-delta (native chunk attention), conditioning-math (native gate and normalization), conditioning-attention (native math SDPA), conditioning-rotary (native BF16 RoPE), or conditioning-l2 (native BF16 normalization); denoiser and sampling use native conditioning")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *model == "" || *references == "" || flags.NArg() != 0 {
		return fmt.Errorf("MOSS transformer requires -model and -references with no positional arguments")
	}
	return latentimage.VerifyMOSSTransformer(context.Background(), *model, *references, *contract, *stage, flags.Output())
}
