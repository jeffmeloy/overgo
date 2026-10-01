package main

import (
	"context"
	"flag"
	"fmt"
	"path/filepath"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/speechsynth"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// speechTrace is the part of a recorded synthesis trace the speech route
// trains on: the text, the noise seed, and the normalized codec latent each
// quantizer call received, one frame per call.
type speechTrace struct {
	Text           string `json:"text"`
	Seed           int64  `json:"seed"`
	QuantizerCalls []struct {
		In struct {
			Values []float32 `json:"values"`
		} `json:"in"`
	} `json:"quantizer_calls"`
}

// trainSpeech is the speech route: under a supervised session it trains
// the joint backbone and flow head of a speech synthesis checkpoint,
// teacher-forced, to reproduce the codec latents a recorded synthesis trace
// holds for its text under its noise seed, refuses a run whose loss on that
// trace does not descend, and publishes the trained model as a checkpoint.
func trainSpeech(ctx context.Context, flags *flag.FlagSet, args []string) (err error) {
	model := flags.String("model", "", "speech synthesis model directory (config + safetensors + tokenizer.model)")
	tracePath := flags.String("trace", "", "recorded synthesis trace JSON (text, seed, quantizer_calls[].in.values); it grounds the recipe dataset")
	steps := clioptions.IntOverride(flags, "steps", "required observed Muon steps")
	maxWall := clioptions.DurationOverride(flags, "max-wall", "optional projected-wall bound")
	storePath := flags.String("store", "overgodb-store", "OvergoDB for the session observation, recipe authority and checkpoint record")
	out := flags.String("out", "", "directory the trained model's checkpoint publishes to")
	if err := flags.Parse(args); err != nil {
		return err
	}
	name := flags.Name()
	if *model == "" || *tracePath == "" || *out == "" || *steps <= 0 {
		return fmt.Errorf("%s: model, trace, out, and positive steps required", name)
	}
	var trace speechTrace
	if err := jsonfile.Decode(*tracePath, &trace); err != nil {
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
	// The trainer steps latent-sequence regression from the trace's text to
	// its codec latents, the audio the model synthesizes.
	session, err := openSessionAs(ctx, store, weights, *tracePath,
		generationContract(trainingprogram.ObjectiveLatentSequence, recipecontract.ModalityText, recipecontract.ModalityAudio, trainingprogram.MetricAudioSNR))
	if err != nil {
		return err
	}
	defer session.settle(ctx, &err)
	loaded, err := speechsynth.Load(directory)
	if err != nil {
		return err
	}
	tokenizer, err := speechsynth.LoadTokenizer(filepath.Join(directory, "tokenizer.model"))
	if err != nil {
		return err
	}
	ids, err := tokenizer.Encode(trace.Text)
	if err != nil {
		return err
	}
	example := speechsynth.TrainingExample{TextIDs: ids, Frames: len(trace.QuantizerCalls), Seed: trace.Seed}
	for _, call := range trace.QuantizerCalls {
		example.Latents = append(example.Latents, call.In.Values...)
	}
	if example.Frames == 0 || len(example.Latents) != example.Frames*loaded.Dims.LatentDim {
		return fmt.Errorf("%s: trace holds %d latent values over %d frames, want %d per frame", name, len(example.Latents), example.Frames, loaded.Dims.LatentDim)
	}
	trained, _ := loaded.TrainedTensors(true)
	var parameters int
	for _, values := range trained {
		parameters += len(values)
	}
	plan, err := probeSession(trainingprogram.ObjectiveLatentSequence, *steps, parameters, *maxWall)
	if err != nil {
		return err
	}
	trainer, err := speechsynth.NewJointTrainer(loaded, plan.Optimizer(), false)
	if err != nil {
		return err
	}
	defer trainer.Close()
	config := trainer.Snapshot().Config
	fmt.Printf("trainable parameters=%d frames=%d text_tokens=%d joint_lr=%.4g derived_momentum=%.4f\n",
		parameters, example.Frames, len(ids), config.BaseLearningRate, config.Momentum)
	return boundedRun{name: name, plan: plan, session: session,
		step: func(step int) (optimizer.ObservedStepResult, error) {
			loss, err := trainer.Step(example)
			return optimizer.ObservedStepResult{
				Loss: loss, Step: step + 1, LearningRate: config.LearningRate(step + 1), GradientL2: trainer.GradientL2(),
			}, err
		},
		loss: func() (float64, error) {
			return loaded.LossAndGrads(example.TextIDs, example.Latents, example.Frames, example.Seed, nil)
		},
		// Every update trains on the one trace.
		publish: func(ctx context.Context) error {
			return session.publishPlanned(ctx, *out, trainer, plan.Updates())
		},
	}.train(ctx)
}
