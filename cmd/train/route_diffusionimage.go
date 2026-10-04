package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/diffusionimage"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// otSigmaMin is the flow path's terminal noise floor, the reference
// trainer's sigma_min.
const otSigmaMin = 0.001

// trainDiffusionImage is the diffusionimage route: under a supervised
// session it trains every tensor of a flow-matching image checkpoint on the
// images a manifest names, all of them one batch, along one seeded
// optimal-transport path per run, refuses a run whose loss on that path does
// not descend, and publishes the trained model as a checkpoint.
func trainDiffusionImage(ctx context.Context, flags *flag.FlagSet, args []string) (err error) {
	model := flags.String("model", "", "flow-matching image model directory (config.json + safetensors)")
	manifest := flags.String("images", "", "UTF-8 manifest naming one training image per line, relative to the manifest; it grounds the recipe dataset")
	steps := clioptions.IntOverride(flags, "steps", "required observed Muon steps")
	maxWall := clioptions.DurationOverride(flags, "max-wall", "optional projected-wall bound")
	storePath := flags.String("store", "overgodb-store", "OvergoDB for the session observation, recipe authority and checkpoint record")
	out := flags.String("out", "", "directory the trained model's checkpoint publishes to")
	if err := flags.Parse(args); err != nil {
		return err
	}
	name := flags.Name()
	if *model == "" || *manifest == "" || *out == "" || *steps <= 0 {
		return fmt.Errorf("%s: model, images, out, and positive steps required", name)
	}
	batch, err := manifestImages(*manifest)
	if err != nil {
		return err
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	directory := roots.ResolveModelPath(*model)
	weights, err := trainingworkflow.IdentifyModelWeights(directory)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	// The trainer steps flow matching from the manifest's images to images.
	session, err := openSessionAs(ctx, store, weights, *manifest,
		generationContract(trainingprogram.ObjectiveFlowMatching, recipecontract.ModalityImage, recipecontract.ModalityImage, trainingprogram.MetricImagePSNRUnit))
	if err != nil {
		return err
	}
	defer session.settle(ctx, &err)
	loaded, err := diffusionimage.Load(directory)
	if err != nil {
		return err
	}
	plan, err := probeSession(trainingprogram.ObjectiveFlowMatching, *steps, loaded.TrainableParameterCount(), *maxWall)
	if err != nil {
		return err
	}
	seed, err := plan.Seed("flow-path")
	if err != nil {
		return err
	}
	config := plan.Optimizer()
	trainer, err := diffusionimage.NewTrainer(loaded, config)
	if err != nil {
		return err
	}
	defer trainer.Close()
	fmt.Printf("trainable parameters=%d images=%d derived_lr=%.4g derived_momentum=%.4f\n",
		loaded.TrainableParameterCount(), len(batch.Examples), config.BaseLearningRate, config.Momentum)
	mean := func(losses []float64, err error) (float64, error) {
		var total float64
		for _, loss := range losses {
			total += loss
		}
		return total / float64(max(len(losses), 1)), err
	}
	return boundedRun{name: name, plan: plan, session: session,
		step: func(step int) (optimizer.ObservedStepResult, error) {
			loss, err := mean(trainer.TrainOTBatch(batch, otSigmaMin, seed))
			return optimizer.ObservedStepResult{
				Loss: loss, Step: step + 1, LearningRate: config.LearningRate(step + 1), GradientL2: trainer.GradientL2(),
			}, err
		},
		loss: func() (float64, error) { return mean(trainer.OTLoss(batch, otSigmaMin, seed)) },
		// Every update trains on the one batch the manifest declares.
		publish: func(ctx context.Context) error {
			return session.publishPlanned(ctx, *out, trainer, plan.Updates())
		},
	}.train(ctx)
}

// manifestImages reads the images a manifest names, one per line relative to
// the manifest, as one batch of image targets.
func manifestImages(manifest string) (trainingdata.Batch, error) {
	raw, err := os.ReadFile(manifest)
	if err != nil {
		return trainingdata.Batch{}, err
	}
	process := trainingdata.ImageProcessor(trainingdata.RoleTarget)
	var batch trainingdata.Batch
	for line := range bytes.Lines(raw) {
		name := string(bytes.TrimSpace(line))
		if name == "" {
			continue
		}
		data, err := os.ReadFile(filepath.Join(filepath.Dir(manifest), name))
		if err != nil {
			return trainingdata.Batch{}, err
		}
		example, err := process(context.Background(), trainingdata.RawRecord{ID: name, Group: name, Data: data})
		if err != nil {
			return trainingdata.Batch{}, fmt.Errorf("image %s: %w", name, err)
		}
		batch.Examples = append(batch.Examples, example)
	}
	if len(batch.Examples) == 0 {
		return trainingdata.Batch{}, fmt.Errorf("manifest %s names no images", manifest)
	}
	return batch, nil
}
