package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"

	"overgo/internal/adaptertrain"
	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/cuda/device"
	"overgo/internal/gguf"
	"overgo/internal/overgodb"
	"overgo/internal/tokenizer"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// The adapter route steps through the CUDA worker, which only the Windows
// build carries, so it registers itself here.
func init() { trainRoutes["adapter"] = trainAdapter }

// adapterTraining is what the adapter route reports: the layer trained, the
// example rows, the adapter's parameters, every step's loss and the
// checkpoint the trained adapter published.
type adapterTraining struct {
	Layer      uint32      `json:"layer"`
	Rows       int         `json:"rows"`
	Parameters int         `json:"parameters"`
	Losses     []float64   `json:"losses"`
	Checkpoint artifact.ID `json:"checkpoint"`
}

// trainAdapter is the adapter route: bounded artifact-declared adapter
// training of one layer of a GGUF model on a teacher-forced text dataset, as
// a supervised session whose trained adapter publishes as a checkpoint.
func trainAdapter(ctx context.Context, flags *flag.FlagSet, args []string) (err error) {
	modelPath := flags.String("model", "", "GGUF model")
	datasetPath := flags.String("dataset", "", "UTF-8 teacher-forced text dataset")
	layer := flags.Uint("layer", 0, "adapter layer")
	steps := clioptions.IntOverride(flags, "steps", "required update steps")
	out := flags.String("out", "", "directory the trained adapter's checkpoint publishes to")
	storePath := flags.String("store", "overgodb-store", "OvergoDB for the session observation, recipe authority and checkpoint record")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *modelPath == "" || *datasetPath == "" || *out == "" || *steps <= 0 || uint64(*layer) > uint64(^uint32(0)) {
		return fmt.Errorf("%s: model, dataset, out, valid layer, and positive steps required", flags.Name())
	}
	store, err := overgodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	session, err := openSession(ctx, store, *modelPath, *datasetPath)
	if err != nil {
		return err
	}
	defer session.settle(ctx, &err)
	ids, err := datasetTokens(*modelPath, session.data)
	if err != nil {
		return err
	}
	input, target, err := trainingdata.AdjacentTokenRows(ids)
	if err != nil {
		return err
	}
	trained, _, err := adaptertrain.LoadArtifact(ctx, *modelPath, uint32(*layer), session.authority.Optimizer)
	if err != nil {
		return err
	}
	plan, err := trainingprogram.CompileProbeSessionPlan(trainingprogram.ProbeSpec{
		Objective: session.authority.Objective, Updates: *steps, MaximumSequence: len(ids),
		Parameters: trained.ParameterCount(), Optimizer: session.authority.Optimizer,
	})
	if err != nil {
		return err
	}
	example, err := trained.BuildExample(ctx, *modelPath, nil, input, target, "text")
	if err != nil {
		return err
	}
	worker, err := device.New(device.DefaultOrdinal())
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, worker.Close()) }()
	result := adapterTraining{Layer: uint32(*layer), Rows: example.Rows, Parameters: trained.ParameterCount(), Losses: make([]float64, plan.Updates())}
	runErr := func() error {
		for index := range result.Losses {
			if err := ctx.Err(); err != nil {
				return err
			}
			loss, err := trained.Step(worker, example)
			if err != nil {
				return err
			}
			result.Losses[index] = loss
		}
		// The dataset is the one record every update trains on, so the run
		// consumed as many records as it took updates.
		stream, err := trainingworkflow.TextStream(session.data, uint64(plan.Updates()))
		if err != nil {
			return err
		}
		checkpoint, err := session.publish(ctx, *out, trained.Program(), trained.Snapshot().Optimizer, stream, func(stage string) error {
			return trained.SaveWeights(adaptertrain.CheckpointWeightsPath(stage))
		})
		result.Checkpoint = checkpoint.ID()
		return err
	}()
	if err := session.finish(ctx, plan, runErr); err != nil {
		return err
	}
	return clioptions.WritePrettyJSON(os.Stdout, result)
}

// datasetTokens encodes a text dataset, specials added, with the tokenizer
// the GGUF artifact at modelPath declares.
func datasetTokens(modelPath string, text []byte) ([]tokenizer.TokenID, error) {
	file, err := gguf.Open(modelPath)
	if err != nil {
		return nil, err
	}
	vocabulary, loadErr := tokenizer.Load(file)
	if err := errors.Join(loadErr, file.Close()); err != nil {
		return nil, err
	}
	return vocabulary.Encode(string(text), tokenizer.EncodeOptions{AddSpecial: true})
}
