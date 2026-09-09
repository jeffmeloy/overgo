package speechrecognition

import (
	"context"
	"errors"

	"overgo/internal/adaptertrain"
	"overgo/internal/recipe"
	"overgo/internal/trainingprogram"
)

// NewOutputAdapter reserves a CPU CTC adapter in this lease's encoder workspace.
// The sample bound comes from the dataset's admission policy; the target bound
// is the resulting maximum CTC output length. The caller must keep this lease
// alive and use the adapter synchronously with PrepareCTC until releasing it.
// Only a standalone, unadapted base recipe is an admissible training source.
func (lease *SpeechLease) NewOutputAdapter(ctx context.Context, maximumSamples uint64, policy trainingprogram.OptimizerPolicy) (*adaptertrain.LinearCTC, error) {
	if lease == nil || ctx == nil {
		return nil, errors.New("transcription training: incomplete adapter request")
	}
	lease.mu.Lock()
	defer lease.mu.Unlock()
	component, err := lease.trainingComponent()
	if err != nil {
		return nil, err
	}
	model := component.transcriber
	frames, err := model.frontend.GroupedFrames(maximumSamples, model.profile.Grouping)
	if err != nil {
		return nil, err
	}
	return model.encoder.NewOutputAdapter(ctx, &component.text.Encoder, frames, model.encoder.outputFrames(frames), model.profile.BlankToken, policy)
}

// PrepareCTC executes the same grouped frontend and frozen encoder as inference,
// then encodes the explicitly transformed training target. Hidden features borrow
// the lease workspace until its next invocation; target text never enters the
// encoder. The caller owns signal admission and the unmodified source record.
func (lease *SpeechLease) PrepareCTC(ctx context.Context, samples []float32, rate int, target string) (adaptertrain.LinearCTCExample, error) {
	if lease == nil || ctx == nil {
		return adaptertrain.LinearCTCExample{}, errors.New("transcription training: incomplete example")
	}
	lease.mu.Lock()
	defer lease.mu.Unlock()
	component, err := lease.trainingComponent()
	if err != nil {
		return adaptertrain.LinearCTCExample{}, err
	}
	if component.text.Encoder.adapter == nil {
		return adaptertrain.LinearCTCExample{}, errors.New("transcription training: adapter was not admitted")
	}
	model := component.transcriber
	features, frames, _, err := model.frontend.ProcessGrouped(ctx, samples, rate, &component.text.Frontend, model.profile.Grouping)
	if err != nil {
		return adaptertrain.LinearCTCExample{}, err
	}
	hidden, frames, err := model.encoder.Encode(ctx, features, frames, &component.text.Encoder, nil)
	if err != nil {
		return adaptertrain.LinearCTCExample{}, err
	}
	targets, err := model.tokenizer.Encode(target)
	if err != nil {
		return adaptertrain.LinearCTCExample{}, err
	}
	return adaptertrain.LinearCTCExample{Hidden: hidden, Frames: frames, Targets: targets}, nil
}

func (lease *SpeechLease) trainingComponent() (*transcriptionComponent, error) {
	if lease.released || len(lease.components) != 1 || lease.definition.Task != recipe.TaskTranscription {
		return nil, errors.New("transcription training: released or composite lease")
	}
	if _, adapted := lease.definition.PrimaryDependency(recipe.DependencyCheckpoint); adapted {
		return nil, errors.New("transcription training: resume requires the base recipe and exact checkpoint owner")
	}
	component := lease.components[0].Model()
	if component == nil || component.transcriber == nil {
		return nil, errors.New("transcription training: recognizer absent")
	}
	if component.transcriber.transducer != nil {
		return nil, errors.New("transcription training: CTC adapter does not support recurrent decoding")
	}
	return component, nil
}
