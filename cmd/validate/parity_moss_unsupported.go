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
	flags.String("stage", "refinement", "native transformer boundary: refinement, joint (first block), denoiser (complete forward), sampling (complete trajectory), decoder (pixels), conditioning-delta (native chunk attention), conditioning-math (native gate and normalization), conditioning-attention (native math SDPA), conditioning-rotary (native BF16 RoPE), conditioning-l2 (native BF16 normalization), or conditioning-model (shared text encoder layers); denoiser and sampling use native conditioning")
	if err := flags.Parse(args); err != nil {
		return err
	}
	return fmt.Errorf("MOSS refinement verification requires Windows CUDA")
}
