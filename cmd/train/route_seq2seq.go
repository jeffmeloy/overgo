package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"strconv"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/seq2seq"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// trainSeq2Seq is the seq2seq route: under a supervised session it trains
// every parameter of an encoder-decoder checkpoint, teacher-forced, on the
// leading records of a JSON Lines dataset of text pairs, one pair per update
// cycling through them, refuses a run whose loss over those pairs does not
// descend, and publishes the trained model as a checkpoint.
func trainSeq2Seq(ctx context.Context, flags *flag.FlagSet, args []string) (err error) {
	model := flags.String("model", "", "encoder-decoder model directory (config.json + safetensors + tokenizer.model)")
	datasetPath := flags.String("dataset", "", "JSON Lines dataset of text pairs, one record per line")
	inputField := flags.String("input-field", "", "the JSON field of each record the encoder reads")
	targetField := flags.String("target-field", "", "the JSON field of each record the decoder learns to write")
	steps := clioptions.IntOverride(flags, "steps", "required observed Muon steps")
	maxWall := clioptions.DurationOverride(flags, "max-wall", "optional projected-wall bound")
	storePath := flags.String("store", "overgodb-store", "OvergoDB for the session observation, recipe authority and checkpoint record")
	out := flags.String("out", "", "directory the trained model's checkpoint publishes to")
	if err := flags.Parse(args); err != nil {
		return err
	}
	name := flags.Name()
	if *model == "" || *datasetPath == "" || *inputField == "" || *targetField == "" || *out == "" || *steps <= 0 {
		return fmt.Errorf("%s: model, dataset, input-field, target-field, out, and positive steps required", name)
	}
	process, err := trainingdata.JSONTextPairProcessor(*inputField, *targetField)
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
	session, err := openSession(ctx, store, weights, *datasetPath)
	if err != nil {
		return err
	}
	defer session.settle(ctx, &err)
	generator, err := seq2seq.LoadGenerator(directory)
	if err != nil {
		return err
	}
	raw := session.data
	// The run trains on at most one record per update; the loss the run is
	// held to is the mean over exactly those records.
	var pairs []seq2seq.TrainingPair
	for line := range bytes.Lines(raw) {
		if len(pairs) == *steps {
			break
		}
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		id := *datasetPath + "/" + strconv.Itoa(len(pairs))
		example, err := process(ctx, trainingdata.RawRecord{ID: id, Group: id, Data: line})
		if err != nil {
			return err
		}
		pair, err := generator.TrainingPair(example)
		if err != nil {
			return err
		}
		pairs = append(pairs, pair)
	}
	if len(pairs) == 0 {
		return fmt.Errorf("%s: dataset %s holds no records", name, *datasetPath)
	}
	var sequence int
	for _, pair := range pairs {
		sequence = max(sequence, len(pair.Source)+len(pair.Targets))
	}
	trainedModel := generator.Model()
	parameters := trainedModel.TrainableParameterCount()
	runPlan, err := trainingprogram.CompileProbeSessionPlan(trainingprogram.ProbeSpec{
		Objective: session.authority.Objective, Updates: *steps, MaximumSequence: sequence,
		MaxProjectedWall: *maxWall, Parameters: parameters, Optimizer: session.authority.Optimizer,
	})
	if err != nil {
		return err
	}
	config := runPlan.Optimizer()
	trainer, err := seq2seq.NewTrainer(trainedModel, config)
	if err != nil {
		return err
	}
	defer trainer.Close()
	fmt.Printf("trainable parameters=%d records=%d derived_lr=%.4g derived_momentum=%.4f\n",
		parameters, len(pairs), config.BaseLearningRate, config.Momentum)
	return boundedRun{name: name, plan: runPlan, session: session,
		step: func(step int) (optimizer.ObservedStepResult, error) {
			loss, err := trainer.Step(pairs[step%len(pairs)])
			return optimizer.ObservedStepResult{
				Loss: loss, Step: step + 1, LearningRate: config.LearningRate(step + 1), GradientL2: trainer.GradientL2(),
			}, err
		},
		loss: func() (float64, error) {
			var total float64
			for _, pair := range pairs {
				loss, err := trainedModel.Loss(pair)
				if err != nil {
					return 0, err
				}
				total += loss
			}
			return total / float64(len(pairs)), nil
		},
		// Every update consumed one record of the dataset.
		publish: func(ctx context.Context) error {
			stream, err := trainingworkflow.TextStream(raw, uint64(runPlan.Updates()))
			if err != nil {
				return err
			}
			_, err = session.publishWeights(ctx, *out, trainer.Program(), trainer, stream)
			return err
		},
	}.train(ctx)
}
