package speechsynth

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/processmeasure"
	"overgo/internal/recipecontract"
	"overgo/internal/workflowruntime"
)

// Speak reads a text aloud in a voice, as a document is read: the document plan
// splits the text into segments with the native frame budget of each, each
// segment is generated and decoded, and their samples follow one another. The
// recording is complete only when every segment ended on its own.
func (s *Synthesizer) Speak(ctx context.Context, request DocumentRequest) (Audio, error) {
	return collectSpeech(func(emit func(Audio) error) (Audio, error) {
		return s.SpeakStream(ctx, request, func(chunk Audio) error {
			if err := media.EmitAudio(ctx, media.DecodedAudio{Samples: chunk.PCM, Format: recipecontract.AudioFormat{SampleRate: uint64(chunk.SampleRate), Channels: uint32(chunk.Channels), Encoding: "pcm-f32le"}}); err != nil {
				return err
			}
			return emit(chunk)
		})
	})
}

func collectSpeech(stream func(func(Audio) error) (Audio, error)) (Audio, error) {
	var pcm []float32
	spoken, err := stream(func(chunk Audio) error { pcm = append(pcm, chunk.PCM...); return nil })
	if err != nil {
		return Audio{}, err
	}
	spoken.PCM = pcm
	return spoken, nil
}

// SpeakStream emits ordered PCM chunks with synchronous backpressure.
// Chunks are provisional and the returned metadata establishes completion.
// The callback owns each chunk's samples and an error stops generation.
func (s *Synthesizer) SpeakStream(ctx context.Context, request DocumentRequest, emit func(Audio) error) (Audio, error) {
	return s.speakStream(ctx, request, seededNoise, emit)
}

func (s *Synthesizer) speakStream(ctx context.Context, request DocumentRequest, noise func(generationPlan) frameNoise, emit func(Audio) error) (Audio, error) {
	if emit == nil {
		return Audio{}, errors.New("speechsynth: streaming requires a chunk consumer")
	}
	// The whole text's parts are named for a ledger that reads them (processmeasure.Stage, served-parts), as a clip's
	// are: the text planned, and for each segment its text tokenized, its voice read and prompted, and its frames
	// generated, each frame's decode and its handing on parts of their own inside that.
	planning := processmeasure.Stage(ctx, "speech text planned")
	plan, err := s.PlanDocument(ctx, request)
	planning()
	if err != nil {
		return Audio{}, err
	}
	var spoken Audio
	spoken.Complete = true
	samples := 0
	for _, segment := range plan.Segments {
		tokenizing := processmeasure.Stage(ctx, "speech text tokenized")
		generation, err := s.tokenize(documentSegmentRequest(plan, segment))
		tokenizing()
		if err != nil {
			return Audio{}, err
		}
		reading := processmeasure.Stage(ctx, "speech voice read")
		voice, params, err := s.prepareGeneration(generation, noise(generation))
		reading()
		if err != nil {
			return Audio{}, err
		}
		prompting := processmeasure.Stage(ctx, "speech voice prompted")
		state, err := s.model.promptVoice(ctx, voice, generation.tokens, params)
		prompting()
		if err != nil {
			return Audio{}, err
		}
		decoder := newCodecState(s.model.Codec)
		spoken.SampleRate, spoken.Channels = s.model.Codec.SampleRate, 1
		generating := processmeasure.Stage(ctx, "speech frames generated")
		complete, err := s.model.walkFrames(ctx, state, params, func(frame []float32, _ float64) error {
			decoding := processmeasure.Stage(ctx, "speech frame decoded")
			pcm, err := s.model.decodeLatents(ctx, decoder, LatentBatch{Values: frame, Frames: 1, Width: s.model.Dims.LatentDim})
			decoding()
			if err != nil {
				return err
			}
			samples += len(pcm)
			defer processmeasure.Stage(ctx, "speech frame handed on")()
			return emit(Audio{PCM: pcm, SampleRate: spoken.SampleRate, Channels: spoken.Channels})
		})
		generating()
		if err != nil {
			return Audio{}, err
		}
		spoken.Complete = spoken.Complete && complete
	}
	if samples == 0 {
		return Audio{}, errors.New("speechsynth: the text produced no samples")
	}
	return spoken, nil
}

func documentSegmentRequest(plan DocumentPlan, part DocumentSegment) SynthesisRequest {
	request := plan.Request.synthesis()
	request.Text, request.MaxFrames = request.Text[part.Start:part.End], part.MaxFrames
	return request
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
			return DocumentRequest{Text: request.Text, Voice: request.Voice, Seed: request.Seed, Temperature: request.Temperature,
				PreparedVoice: request.PreparedVoice, ReferenceAudio: request.ReferenceAudio, ReferenceStartSample: request.ReferenceStartSample,
				ReferenceEndSample: request.ReferenceEndSample, VoiceMemoryBytes: request.VoiceMemoryBytes}, nil
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
