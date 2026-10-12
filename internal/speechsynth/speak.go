package speechsynth

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/workflowruntime"
)

// Speak reads a text aloud in a voice, as a document is read: the document plan
// splits the text into segments with the native frame budget of each, each
// segment is generated and decoded, and their samples follow one another. The
// recording is complete only when every segment ended on its own.
func (s *Synthesizer) Speak(ctx context.Context, request DocumentRequest) (Audio, error) {
	return s.speakFrom(ctx, request, seededNoise)
}

// speakFrom is Speak with the noise of each segment's generation given by the
// caller: a served request draws it from its seed, and a comparison with the
// model's own code hands in the noise that code drew.
func (s *Synthesizer) speakFrom(ctx context.Context, request DocumentRequest, noise func(generationPlan) frameNoise) (Audio, error) {
	plan, err := s.PlanDocument(ctx, request)
	if err != nil {
		return Audio{}, err
	}
	var spoken Audio
	spoken.Complete = true
	for _, segment := range plan.Segments {
		generation, err := s.tokenize(documentSegmentRequest(plan, segment))
		if err != nil {
			return Audio{}, err
		}
		latents, err := s.generateFrom(ctx, generation, noise(generation))
		if err != nil {
			return Audio{}, err
		}
		audio, err := s.decode(ctx, latents)
		if err != nil {
			return Audio{}, err
		}
		if spoken.SampleRate != 0 && (audio.SampleRate != spoken.SampleRate || audio.Channels != spoken.Channels) {
			return Audio{}, fmt.Errorf("speechsynth: a segment's format differs from the first: %d Hz, %d channels against %d, %d", audio.SampleRate, audio.Channels, spoken.SampleRate, spoken.Channels)
		}
		spoken.SampleRate, spoken.Channels = audio.SampleRate, audio.Channels
		spoken.PCM = append(spoken.PCM, audio.PCM...)
		spoken.Complete = spoken.Complete && audio.Complete
	}
	if len(spoken.PCM) == 0 {
		return Audio{}, errors.New("speechsynth: the text produced no samples")
	}
	return spoken, nil
}

func documentSegmentRequest(plan DocumentPlan, part DocumentSegment) SynthesisRequest {
	return SynthesisRequest{Text: plan.Request.Text[part.Start:part.End], Voice: plan.Request.Voice, Seed: plan.Request.Seed, Temperature: plan.Request.Temperature, MaxFrames: part.MaxFrames}
}

// RegisterWholeText binds one synthesizer to the speech recipe's three stages
// as a reading of the whole text, for a request that names no frame limit. The
// stages keep the recipe's names and carry other things than a preview's:
// tokenize hands on the text and its settings, generate is Speak, which
// generates and decodes every segment, and decode only refuses a reading that
// was cut. The time generate records is therefore the whole reading's, and
// the other two record almost none. Only the WAV is stored, since a long
// reading's samples as a JSON list pass what one stored document may hold: an
// answered request is replayed, and any other is read again from its start.
func RegisterWholeText(runtime *workflowruntime.Runtime, modelID artifact.ID, synthesizer *Synthesizer) error {
	if synthesizer == nil {
		return errors.New("speechsynth: incomplete runtime binding")
	}
	return registerWholeText(runtime, modelID, synthesizer.Speak)
}

// registerWholeText is RegisterWholeText with the reading given by the caller,
// so the binding's rules are held without a model.
func registerWholeText(runtime *workflowruntime.Runtime, modelID artifact.ID, speak func(context.Context, DocumentRequest) (Audio, error)) error {
	return workflowruntime.RegisterContextPipeline(runtime, modelID,
		modelrecipe.ModuleSpeechTokenize, func(_ context.Context, request SynthesisRequest) (DocumentRequest, error) {
			return DocumentRequest{Text: request.Text, Voice: request.Voice, Seed: request.Seed, Temperature: request.Temperature}, nil
		},
		modelrecipe.ModuleSpeechGenerate, speak,
		// A whole reading is not published short. Every segment has been read by
		// now, and the same request is cut the same way again, so the refusal
		// says what to change.
		modelrecipe.ModuleSpeechDecode, func(_ context.Context, audio Audio) (Audio, error) {
			if !audio.Complete {
				return Audio{}, errors.New("speechsynth: the reading did not finish by itself and is not published cut short; change the seed or the temperature")
			}
			return audio, nil
		}, WAVContent)
}
