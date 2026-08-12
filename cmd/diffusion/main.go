package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/inference"
)

type cliConfig struct {
	model      string
	repository string
	prompt     string
	device     int
	lora       []string
	visual     bool
	diffusion  inference.DiffusionOptions
}

func main() {
	clioptions.Main(func() error { return run(os.Args[1:], os.Stdout, os.Stderr) })
}

func run(arguments []string, stdout, stderr io.Writer) error {
	config, err := parseCLI(arguments)
	if err != nil {
		return err
	}
	runner, err := clioptions.OpenRunner(
		context.Background(), config.repository, config.model,
		clioptions.BuildOpenOptions(config.device, config.lora, 1),
	)
	if err != nil {
		return err
	}
	defer runner.Close()
	if config.visual {
		config.diffusion.OnStep = func(step inference.DiffusionStep) error {
			_, err := fmt.Fprintf(stderr, "\rdiffusion step %d/%d", step.Step+1, step.TotalSteps)
			return err
		}
	}
	_, text, err := runner.GenerateDiffusion(context.Background(), config.prompt, config.diffusion)
	if config.visual {
		fmt.Fprintln(stderr)
	}
	if err != nil {
		return err
	}
	_, err = fmt.Fprintln(stdout, text)
	return err
}

func parseCLI(arguments []string) (cliConfig, error) {
	flags := flag.NewFlagSet("diffusion", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	modelFlags := clioptions.AddModelFlags(flags, "GGUF LoRA adapter at scale 1; repeatable")
	length := flags.Int("length", 512, "total prompt-plus-output sequence length")
	steps := flags.Int("steps", 128, "diffusion step count")
	algorithm := flags.Int("algorithm", 4, "ranking: 0 origin, 1 entropy, 2 margin, 3 random, 4 confidence")
	epsilon := flags.Float64("eps", 0, "timestep schedule epsilon")
	blockLength := flags.Int("block-length", 0, "block schedule length")
	cfgScale := flags.Float64("cfg-scale", 0, "classifier-free guidance scale")
	algorithmTemperature := flags.Float64("alg-temp", 0, "probabilistic ranking temperature")
	gumbel := flags.Bool("gumbel", false, "apply pinned logit-space Gumbel transform")
	temperature := flags.Float64("temp", 0.8, "token sampling temperature")
	topK := flags.Int("top-k", 40, "top-k candidate count; zero disables")
	topP := flags.Float64("top-p", 0.95, "nucleus sampling probability")
	seed := flags.Int64("seed", 0, "sampling seed")
	shiftLogits := flags.String("shift-logits", "auto", "logit alignment: auto, true, or false")
	visual := flags.Bool("visual", false, "show progressive step count")
	if err := flags.Parse(arguments); err != nil {
		return cliConfig{}, err
	}
	if flags.NArg() != 2 {
		return cliConfig{}, errors.New("usage: diffusion [options] <model.gguf> <prompt>")
	}
	if (*epsilon == 0) == (*blockLength == 0) {
		return cliConfig{}, errors.New("diffusion: set exactly one of -eps or -block-length")
	}
	if *length < 1 || *steps < 1 || *algorithm < 0 || *algorithm > 4 ||
		*topK < 0 || *topP <= 0 || *topP > 1 || *epsilon < 0 || *epsilon > 1 || *blockLength < 0 ||
		*temperature < 0 || *cfgScale < 0 || *algorithmTemperature < 0 ||
		math.IsNaN(*temperature) || math.IsInf(*temperature, 0) ||
		math.IsNaN(*topP) || math.IsInf(*topP, 0) || math.IsNaN(*epsilon) || math.IsInf(*epsilon, 0) ||
		math.IsNaN(*cfgScale) || math.IsInf(*cfgScale, 0) ||
		math.IsNaN(*algorithmTemperature) || math.IsInf(*algorithmTemperature, 0) {
		return cliConfig{}, errors.New("diffusion: numeric option is out of range")
	}
	if *blockLength > 0 {
		blocks := *length / *blockLength
		if *length%*blockLength != 0 || blocks == 0 || *steps%blocks != 0 {
			return cliConfig{}, errors.New("diffusion: block length must divide length and block count must divide steps")
		}
	}
	var shift *bool
	switch strings.ToLower(strings.TrimSpace(*shiftLogits)) {
	case "auto":
	case "true":
		value := true
		shift = &value
	case "false":
		value := false
		shift = &value
	default:
		return cliConfig{}, fmt.Errorf("diffusion: invalid -shift-logits value %q", *shiftLogits)
	}
	schedule := inference.DiffusionTimestep
	if *blockLength != 0 {
		schedule = inference.DiffusionBlock
	}
	return cliConfig{
		model: flags.Arg(0), repository: *modelFlags.Repository,
		prompt: flags.Arg(1), device: *modelFlags.DeviceOrdinal,
		lora: modelFlags.LoRAPaths(), visual: *visual,
		diffusion: inference.DiffusionOptions{
			MaxLength: *length, Steps: *steps,
			Temperature: float32(*temperature), TopK: *topK, TopP: float32(*topP), Seed: *seed,
			Algorithm: inference.DiffusionAlgorithm(*algorithm), Schedule: schedule,
			Epsilon: float32(*epsilon), BlockLength: *blockLength, CFGScale: float32(*cfgScale),
			AlgorithmTemperature: float32(*algorithmTemperature), AddGumbelNoise: *gumbel,
			ShiftLogits: shift,
		},
	}, nil
}
