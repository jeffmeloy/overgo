package main

import (
	"context"
	"flag"
	"fmt"
	"time"

	"overgo/internal/checked"
	"overgo/internal/clioptions"
	"overgo/internal/controllertrain"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/scratchmodel"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingworkflow"
)

// The controller route trains through the resident scratch trainer on the
// CUDA device, which only the Windows build carries, so it registers itself
// here.
func init() { trainRoutes["controller"] = trainController }

// trainController is the controller route: under a supervised session it
// derives a scratch model from the training records of a controller corpus,
// trains it on the device one training document per update, judges it on
// the corpus's held-out records before and after, refuses a run whose
// held-out loss does not descend, and publishes the trained model as a
// checkpoint whose initial state is the scratch construction.
func trainController(ctx context.Context, flags *flag.FlagSet, args []string) error {
	corpusPath := flags.String("corpus", "", "controller corpus: a JSON object of train and holdout records")
	seed := flags.Int64("seed", 0, "seed the scratch model splits its documents and draws its initial weights with")
	steps := clioptions.IntOverride(flags, "steps", "required Muon steps")
	storePath := flags.String("store", "overgodb-store", "OvergoDB for the session observation, recipe authority and checkpoint record")
	out := flags.String("out", "", "directory the trained model's checkpoint publishes to")
	if err := flags.Parse(args); err != nil {
		return err
	}
	name := flags.Name()
	if *corpusPath == "" || *out == "" || *steps <= 0 {
		return fmt.Errorf("%s: corpus, out, and positive steps required", name)
	}
	var spec controllertrain.Spec
	if err := jsonfile.DecodeStrict(*corpusPath, &spec); err != nil {
		return err
	}
	corpus, err := controllertrain.Compile(spec)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	profile, err := scratchmodel.PublishDerivationProfileCatalog(ctx, store)
	if err != nil {
		return err
	}
	construction, err := scratchmodel.Compile(scratchmodel.CorpusFacts{Documents: corpus.TrainingDocuments(), Seed: *seed, Steps: *steps}, profile)
	if err != nil {
		return err
	}
	parameters := construction.OptimizerPlan().ParameterCount()
	plan, err := probeSession(construction.Program().Objective(), *steps, parameters, 0)
	if err != nil {
		return err
	}
	session, err := scratchSession(ctx, store, construction.Authority(), construction.Processor(), trainingworkflow.TokenContract)
	if err != nil {
		return err
	}
	fmt.Printf("scratch model=%s parameters=%d training documents=%d seed=%d\n",
		construction.ID(), parameters, len(construction.Split().Train), *seed)
	runErr := func() error {
		// The resident trainer takes its updates in one call, so the run
		// stops only before training and before publication.
		if err := ctx.Err(); err != nil {
			return err
		}
		started := time.Now()
		run, err := controllertrain.TrainSeed(corpus, construction, session.authority.Optimizer, *steps)
		session.observer.Phase(runrecord.PhaseForwardBackward, time.Since(started))
		if err != nil {
			return err
		}
		before, after := run.Evidence.Initial, run.Evidence.Final
		fmt.Printf("held-out: loss %.6f -> %.6f action accuracy %.3f -> %.3f modality accuracy %.3f -> %.3f\n",
			before.Loss, after.Loss, before.ActionAccuracy, after.ActionAccuracy, before.ModalityAccuracy, after.ModalityAccuracy)
		if !checked.Finite64(after.Loss) || !(after.Loss < before.Loss) {
			return fmt.Errorf("%s: held-out loss did not descend: before=%.6f after=%.6f", name, before.Loss, after.Loss)
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		// The run read its split's training documents in order, one per
		// update.
		stream := trainingdata.StreamState{Identity: construction.SplitID(), Position: uint64(*steps)}
		_, err = session.publishWeights(ctx, *out, construction.Program(), run.Trained, stream)
		return err
	}()
	return session.finish(ctx, plan, runErr)
}
