package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/optimizer"
	"overgo/internal/oscillatorimage"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// trainOscillatorImage is the oscillatorimage route: it measures the real
// checkpoint's loss on the evaluation seed, then, under a supervised session
// whose recipe is the scratch bootstrap's, trains that bootstrap of the real
// checkpoint's configuration by latent L2 toward class images, refuses a
// run that does not descend, and publishes the trained bootstrap as a
// checkpoint whose initial state is the scratch construction.
func trainOscillatorImage(ctx context.Context, flags *flag.FlagSet, args []string) (err error) {
	model := flags.String("model", "", "oscillator image model directory (config.json + safetensors)")
	steps := clioptions.IntOverride(flags, "steps", "required observed Muon steps")
	storePath := flags.String("store", "overgodb-store", "OvergoDB for the session observation, recipe authority and checkpoint record")
	stimulus := flags.String("stimulus", "", "committed declaration of the procedural dataset, grounding the recipe dataset")
	out := flags.String("out", "", "directory the trained bootstrap's checkpoint publishes to")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *model == "" || *stimulus == "" || *out == "" || *steps <= 0 {
		return fmt.Errorf("%s: model, stimulus, out, and positive steps required", flags.Name())
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	directory := roots.ResolveModelPath(*model)
	declaration, err := os.ReadFile(*stimulus)
	if err != nil {
		return err
	}
	real, err := oscillatorimage.Load(directory)
	if err != nil {
		return err
	}
	plan, err := probeSession(trainingprogram.ObjectiveLatentL2, *steps, real.TrainableParameterCount(), 0)
	if err != nil {
		return err
	}
	config := plan.Optimizer()
	seed, err := plan.Seed("bootstrap")
	if err != nil {
		return err
	}
	evalSeed, err := plan.Seed("evaluation")
	if err != nil {
		return err
	}
	construction, processor, err := bootstrapConstruction(real.Cfg, real.TrainableParameterCount(), seed, declaration)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	// The trainer regresses each class's decoded image toward its target:
	// latent L2 from the class label to an image.
	session, err := scratchSession(ctx, store, construction, processor,
		generationContract(trainingprogram.ObjectiveLatentL2, recipecontract.ModalityText, recipecontract.ModalityImage, trainingprogram.MetricImagePSNRUnit))
	if err != nil {
		return err
	}
	defer session.settle(ctx, &err)
	session.data = declaration
	realTrainer, err := oscillatorimage.NewTrainer(real, config, evalSeed)
	if err != nil {
		return err
	}
	defer realTrainer.Close()
	shipped := realTrainer.Loss(evalSeed)
	fmt.Printf("objective identity: real checkpoint loss=%.6f (reference shipped band 15.57 +/- 3*1.16) parameters=%d\n",
		shipped, realTrainer.ParameterCount())

	scratch := oscillatorimage.NewBootstrapModel(real.Cfg, seed)
	trainer, err := oscillatorimage.NewTrainer(scratch, config, seed)
	if err != nil {
		return err
	}
	defer trainer.Close()
	return boundedRun{name: flags.Name(), plan: plan, session: session,
		step: func(int) (optimizer.ObservedStepResult, error) {
			result, err := trainer.Step()
			return optimizer.ObservedStepResult(result), err
		},
		loss: func() (float64, error) { return trainer.Loss(evalSeed), nil },
		// Every update draws one fresh record of the declared procedural
		// dataset.
		publish: func(ctx context.Context) error {
			return session.publishPlanned(ctx, *out, trainer, plan.Updates())
		},
	}.train(ctx)
}

// bootstrapConstruction is the scratch construction an oscillator image
// bootstrap starts from: config's topology, its parameters, the normal draw
// seed initializes them with, and the procedural dataset the declaration
// grounds, which seed also samples. It returns the processor that reads the
// declaration.
func bootstrapConstruction(config oscillatorimage.Config, parameters int, seed int64, declaration []byte) (trainingprogram.ScratchConstruction, artifact.ID, error) {
	dataset, split, processor, err := trainingworkflow.TextAuthority(declaration)
	if err != nil {
		return trainingprogram.ScratchConstruction{}, artifact.ID{}, err
	}
	identify := func(kind artifact.Kind, value any) artifact.ID {
		if err == nil {
			var id artifact.ID
			id, err = artifact.JSONID(kind, value)
			return id
		}
		return artifact.ID{}
	}
	topology := identify(artifact.KindProfile, struct {
		Version string                 `json:"version"`
		Config  oscillatorimage.Config `json:"config"`
	}{"oscillatorimage-topology-v1", config})
	manifest := identify(artifact.KindTensorInventory, struct {
		Version    string      `json:"version"`
		Topology   artifact.ID `json:"topology"`
		Parameters int         `json:"parameters"`
	}{"oscillatorimage-parameters-v1", topology, parameters})
	initializer := identify(artifact.KindProfile, struct {
		Version string `json:"version"`
		Seed    int64  `json:"seed"`
	}{"oscillatorimage-bootstrap-normal-v1", seed})
	initialized := identify(artifact.KindModel, struct {
		Topology    artifact.ID `json:"topology"`
		Manifest    artifact.ID `json:"manifest"`
		Initializer artifact.ID `json:"initializer"`
	}{topology, manifest, initializer})
	recipeID := identify(artifact.KindRecipe, struct {
		Dataset artifact.ID `json:"dataset"`
		Split   artifact.ID `json:"split"`
		Model   artifact.ID `json:"model"`
	}{dataset, split, initialized})
	derivation := identify(artifact.KindProfile, "oscillatorimage-bootstrap-of-configuration-v1")
	rng := identify(artifact.KindProfile, "go-math-rand-v1")
	if err != nil {
		return trainingprogram.ScratchConstruction{}, artifact.ID{}, err
	}
	stream := uint64(seed)
	construction, err := trainingprogram.CompileScratchConstruction(trainingprogram.ScratchSpec{
		Recipe: recipeID, Dataset: dataset, Split: split, DerivationProfile: derivation, TopologyProfile: topology,
		ParameterManifest: manifest, InitializerProfile: initializer, InitializedModel: initialized,
		RNGStreams: []trainingprogram.RNGStreamSpec{
			{Name: "augmentation", Algorithm: rng, Seed: stream}, {Name: "data", Algorithm: rng, Seed: stream},
			{Name: "init", Algorithm: rng, Seed: stream}, {Name: "split", Algorithm: rng, Seed: stream},
		},
	})
	return construction, processor, err
}
