package main

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/clioptions"
	"overgo/internal/latentvideo"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/rae"
	"overgo/internal/recipecontract"
	"overgo/internal/tensor"
	"overgo/internal/trainingprogram"
	"overgo/internal/vitencoder"
)

// trainCodec is the codec route: under a supervised session it trains the
// part of an installed autoencoder that the existing backward operators
// reach, against the pictures the autoencoder should give back. For a bridged
// pair that part is its bridge: the squared error between each training frame
// and the picture the pair's decoder draws is carried back through the
// frozen decoder, and Muon steps the bridge. The route refuses a run whose
// loss on its first batch does not descend, publishes the trained bridge as a
// checkpoint, and writes it as a bridged pair of its own. A directory whose
// decoder has no backward pass is refused by the name of its first operation.
func trainCodec(ctx context.Context, flags *flag.FlagSet, args []string) (err error) {
	pairDir := flags.String("pair", "", "the bridged pair to train: its directory, beside the encoder and the collection it joins")
	manifest := flags.String("frames", "", "UTF-8 manifest naming one training clip per line, relative to the manifest; each clip's first frame is a training picture, and the manifest grounds the recipe dataset")
	batch := flags.Int("batch", 0, "training pictures a step is taken on")
	steps := clioptions.IntOverride(flags, "steps", "required observed Muon steps")
	maxWall := clioptions.DurationOverride(flags, "max-wall", "optional projected-wall bound")
	cache := flags.String("cache", "", "directory that keeps each frame's tokens between runs")
	storePath := flags.String("store", "overgodb-store", "OvergoDB for the session observation, recipe authority and checkpoint record")
	out := flags.String("out", "", "directory the trained bridge's checkpoint publishes to")
	pairOut := flags.String("pair-out", "", "optional directory, beside the encoder and the collection, to write the trained bridge to as a pair of its own")
	if err := flags.Parse(args); err != nil {
		return err
	}
	name := flags.Name()
	if *pairDir == "" || *manifest == "" || *cache == "" || *out == "" || *steps <= 0 || *batch <= 0 {
		return fmt.Errorf("%s: pair, frames, cache, out, and positive steps and batch required", name)
	}
	resolved, bridged, err := rae.ResolveBridgedPair(*pairDir)
	if err != nil {
		return err
	}
	if !bridged {
		return refuseUntrainedDecoder(name, *pairDir)
	}
	if *pairOut != "" && filepath.Dir(filepath.Clean(*pairOut)) != filepath.Dir(filepath.Clean(*pairDir)) {
		return fmt.Errorf("%s: the trained pair sits beside the pair it was trained from", name)
	}
	clips, err := manifestLines(*manifest)
	if err != nil {
		return err
	}
	if len(clips) < *batch {
		return fmt.Errorf("%s: the manifest names %d clips, fewer than one batch of %d", name, len(clips), *batch)
	}
	store, err := overgodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	// The trainer steps the squared error from frames to the frames the pair
	// gives back through its latent.
	session, err := openSessionAs(ctx, store, resolved.Bridge, *manifest,
		generationContract(trainingprogram.ObjectiveImageLatent, recipecontract.ModalityImage, recipecontract.ModalityImage, trainingprogram.MetricImagePSNRUnit))
	if err != nil {
		return err
	}
	defer session.settle(ctx, &err)
	pair, err := rae.LoadBridgedAutoencoder(*pairDir)
	if err != nil {
		return err
	}
	source, err := vitencoder.Load(resolved.SourceEncoder)
	if err != nil {
		return err
	}
	collection, err := rae.LoadAutoencoder(resolved.CollectionDirectory)
	if err != nil {
		return err
	}
	cacheDirectory := filepath.Join(*cache, filepath.Base(resolved.SourceEncoder)+"--"+filepath.Base(resolved.CollectionDirectory))
	if err := os.MkdirAll(cacheDirectory, bridgeCacheMode); err != nil {
		return err
	}
	examples := make([]rae.BridgeExample, len(clips))
	for index, clip := range clips {
		frame, err := clipBridgeFrame(ctx, collection, source, filepath.Join(filepath.Dir(*manifest), clip), cacheDirectory)
		if err != nil {
			return err
		}
		examples[index] = rae.BridgeExample{SourceTokens: frame.source, Picture: frame.picture}
	}
	plan, err := probeSession(trainingprogram.ObjectiveImageLatent, *steps, pair.BridgeParameterCount(), *maxWall)
	if err != nil {
		return err
	}
	config := plan.Optimizer()
	trainer, err := rae.NewBridgeTrainer(pair, config)
	if err != nil {
		return err
	}
	fmt.Printf("trainable parameters=%d frames=%d batch=%d derived_lr=%.4g derived_momentum=%.4f\n",
		trainer.TrainableParameterCount(), len(examples), *batch, config.BaseLearningRate, config.Momentum)
	// A step's batch is the next run of the manifest's frames, wrapping at its end.
	batchAt := func(step int) []rae.BridgeExample {
		chosen := make([]rae.BridgeExample, *batch)
		for at := range chosen {
			chosen[at] = examples[(step**batch+at)%len(examples)]
		}
		return chosen
	}
	return boundedRun{name: name, plan: plan, session: session,
		step: func(step int) (optimizer.ObservedStepResult, error) {
			loss, err := trainer.Step(ctx, batchAt(step))
			return optimizer.ObservedStepResult{
				Loss: loss, Step: step + tensor.SingletonExtent, LearningRate: config.LearningRate(step + tensor.SingletonExtent), GradientL2: trainer.GradientL2(),
			}, err
		},
		// The run is held to the loss of the first batch, before and after.
		loss: func() (float64, error) { return trainer.Loss(ctx, batchAt(0)) },
		publish: func(ctx context.Context) error {
			if *pairOut != "" {
				if err := trainer.SavePair(*pairOut, resolved.SourceEncoder, resolved.CollectionDirectory); err != nil {
					return err
				}
			}
			return session.publishPlanned(ctx, *out, trainer, plan.Updates())
		},
	}.train(ctx)
}

// refuseUntrainedDecoder says why a directory that is no bridged pair cannot
// be trained by this route: a Wan autoencoder's decoder is refused by the
// name of its first operation, none of whose kind has a backward pass, and
// any other directory as holding nothing the route trains.
func refuseUntrainedDecoder(route, directory string) error {
	if _, err := os.Stat(latentvideo.AutoencoderWeights(directory)); err != nil {
		return fmt.Errorf("%s: %s holds no bridged pair and no autoencoder this route knows", route, directory)
	}
	operations, err := latentvideo.AutoencoderDecoderOperations(directory)
	if err != nil {
		return err
	}
	first, ok := firstOperation(operations)
	if !ok {
		return errors.New(route + ": the autoencoder's decoder compiles to no operations")
	}
	return fmt.Errorf("%s: the autoencoder in %s cannot be trained here: its decoder's first operation, %s, convolves over frames, rows and columns, and none of the decoder's %d operations has a backward pass",
		route, directory, first, len(operations))
}

// firstOperation is the first name of a decoder's operations.
func firstOperation(operations []string) (string, bool) {
	if len(operations) == 0 {
		return "", false
	}
	return operations[0], true
}

// manifestLines reads the names a manifest lists, one per line.
func manifestLines(manifest string) ([]string, error) {
	raw, err := os.ReadFile(manifest)
	if err != nil {
		return nil, err
	}
	var names []string
	for line := range bytes.Lines(raw) {
		if name := string(bytes.TrimSpace(line)); name != "" {
			names = append(names, name)
		}
	}
	if len(names) == 0 {
		return nil, fmt.Errorf("manifest %s names no clips", manifest)
	}
	return names, nil
}
