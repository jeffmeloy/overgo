package speechsynth

import (
	"context"
	"errors"
	"fmt"
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
