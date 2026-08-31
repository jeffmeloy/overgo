package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"overgo/internal/checked"
	"overgo/internal/clioptions"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
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
	policy, found, err := modelrecipe.CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil {
		return cliConfig{}, err
	}
	if !found {
		return cliConfig{}, errors.New("diffusion: inference runtime policy is unavailable")
	}
	defaults := policy.Interactive.Diffusion
	flags := flag.NewFlagSet("diffusion", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	modelFlags := clioptions.AddModelFlags(flags, "GGUF LoRA adapter at scale 1; repeatable")
	length := flags.Int("length", defaults.Length, "total prompt-plus-output sequence length")
	steps := flags.Int("steps", defaults.Steps, "diffusion step count")
	algorithm := flags.Int("algorithm", defaults.Algorithm, "ranking: 0 origin, 1 entropy, 2 margin, 3 random, 4 confidence")
	epsilon := clioptions.Float64Override(flags, "eps", "timestep schedule epsilon")
	blockLength := clioptions.IntOverride(flags, "block-length", "block schedule length")
	cfgScale := clioptions.Float64Override(flags, "cfg-scale", "classifier-free guidance scale")
	algorithmTemperature := clioptions.Float64Override(flags, "alg-temp", "probabilistic ranking temperature")
	gumbel := clioptions.BoolOverride(flags, "gumbel", "apply pinned logit-space Gumbel transform")
	temperature := flags.Float64("temp", float64(defaults.Temperature), "token sampling temperature")
	topK := flags.Int("top-k", defaults.TopK, "top-k candidate count; zero disables")
	topP := flags.Float64("top-p", float64(defaults.TopP), "nucleus sampling probability")
	seed := clioptions.Int64Override(flags, "seed", "sampling seed")
	shiftLogits := flags.String("shift-logits", "auto", "logit alignment: auto, true, or false")
	visual := clioptions.BoolOverride(flags, "visual", "show progressive step count")
	if err := flags.Parse(arguments); err != nil {
		return cliConfig{}, err
	}
	if flags.NArg() != 2 {
		return cliConfig{}, errors.New("usage: diffusion [options] <model.gguf> <prompt>")
	}
	positionals := flags.Args()
	if (*epsilon == 0) == (*blockLength == 0) {
		return cliConfig{}, errors.New("diffusion: set exactly one of -eps or -block-length")
	}
	if *length < 1 || *steps < 1 || *algorithm < 0 || *algorithm > int(inference.DiffusionConfidence) ||
		*topK < 0 || *topP <= 0 || *topP > 1 || *epsilon < 0 || *epsilon > 1 || *blockLength < 0 ||
		*temperature < 0 || *cfgScale < 0 || *algorithmTemperature < 0 ||
		!finite(*temperature) || !finite(*topP) || !finite(*epsilon) ||
		!finite(*cfgScale) || !finite(*algorithmTemperature) {
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
		model: positionals[0], repository: *modelFlags.Repository,
		prompt: positionals[1], device: *modelFlags.DeviceOrdinal,
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

func finite(value float64) bool {
	return checked.Finite64(value)
}
