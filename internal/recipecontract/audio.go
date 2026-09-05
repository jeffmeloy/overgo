package recipecontract

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/textcheck"
)

// AudioFormat is the decoded sample contract declared by an artifact profile.
type AudioFormat struct {
	SampleRate uint64 `json:"sample_rate"`
	Channels   uint32 `json:"channels"`
	Encoding   string `json:"encoding"`
}

// Validate rejects an incomplete or non-canonical decoded sample contract.
func (format AudioFormat) Validate() error {
	if !checked.Nonzero(format.SampleRate) || !checked.Nonzero(format.Channels) ||
		!textcheck.LowerIdentifier(format.Encoding, len(format.Encoding)) {
		return errors.New("audio contract: invalid format")
	}
	return nil
}

// AudioFrameGeometry describes the artifact-declared window and hop in source
// samples. FeatureBins is zero for waveform-only tasks.
type AudioFrameGeometry struct {
	WindowSamples uint64 `json:"window_samples"`
	HopSamples    uint64 `json:"hop_samples"`
	FeatureBins   uint32 `json:"feature_bins,omitzero"`
}

// Validate rejects frame geometry that cannot advance through source samples.
func (geometry AudioFrameGeometry) Validate() error {
	if geometry.WindowSamples == 0 || geometry.HopSamples == 0 {
		return errors.New("audio contract: invalid frame geometry")
	}
	return nil
}

// FrameCount derives the number of complete frames admitted by this geometry.
func (geometry AudioFrameGeometry) FrameCount(samples uint64) (uint64, error) {
	if err := geometry.Validate(); err != nil {
		return 0, err
	}
	if samples < geometry.WindowSamples {
		return 0, nil
	}
	return 1 + (samples-geometry.WindowSamples)/geometry.HopSamples, nil
}

// AudioReference binds audio bytes to their artifact-owned format profile.
type AudioReference struct {
	Audio   artifact.ID `json:"audio"`
	Profile artifact.ID `json:"profile"`
}

// Validate rejects untyped audio and non-profile format authorities.
func (reference AudioReference) Validate() error {
	switch reference.Audio.Kind() {
	case artifact.KindDataset, artifact.KindDatasetShard, artifact.KindOutput, artifact.KindFile:
	default:
		return errors.New("audio contract: invalid audio artifact kind")
	}
	if reference.Profile.Kind() != artifact.KindProfile {
		return errors.New("audio contract: invalid audio reference")
	}
	return nil
}

// SampleSpan is a half-open source-audio interval.
type SampleSpan struct {
	Start uint64 `json:"start"`
	End   uint64 `json:"end"`
}

// Validate rejects empty or reversed source-audio intervals.
func (span SampleSpan) Validate() error {
	if span.End <= span.Start {
		return errors.New("audio contract: invalid sample span")
	}
	return nil
}

// Transcription is text derived from one exact audio artifact.
type Transcription struct {
	Source   AudioReference `json:"source"`
	Text     string         `json:"text"`
	Language string         `json:"language,omitzero"`
}

// Validate rejects an unbound source or non-canonical language tag.
func (transcription Transcription) Validate() error {
	if err := transcription.Source.Validate(); err != nil {
		return err
	}
	if transcription.Language != strings.TrimSpace(transcription.Language) ||
		strings.ContainsAny(transcription.Language, "\r\n") {
		return errors.New("audio contract: invalid transcript language")
	}
	return nil
}

// AlignedText binds one transcript element to a source-audio interval.
type AlignedText struct {
	Span       SampleSpan `json:"span"`
	Text       string     `json:"text"`
	Confidence float64    `json:"confidence"`
}

// TimestampedAlignment carries sample-exact ordered transcript alignment.
type TimestampedAlignment struct {
	Source        AudioReference `json:"source"`
	Transcription artifact.ID    `json:"transcription"`
	Items         []AlignedText  `json:"items"`
}

// Validate rejects dangling transcripts, invalid confidence, and overlapping
// or non-canonical alignment order.
func (alignment TimestampedAlignment) Validate() error {
	if err := alignment.Source.Validate(); err != nil {
		return err
	}
	if alignment.Transcription.Kind() != artifact.KindOutput {
		return errors.New("audio contract: invalid timestamped alignment authority")
	}
	var priorEnd uint64
	for index, item := range alignment.Items {
		if err := item.Span.Validate(); err != nil {
			return fmt.Errorf("audio contract: alignment item %d: %w", index, err)
		}
		if strings.TrimSpace(item.Text) == "" || !checked.UnitInterval64(item.Confidence) ||
			index > 0 && item.Span.Start < priorEnd {
			return fmt.Errorf("audio contract: invalid alignment item %d", index)
		}
		priorEnd = item.Span.End
	}
	return nil
}

// SpeechTurn is a speaker-labelled source-audio interval.
type SpeechTurn struct {
	Span       SampleSpan `json:"span"`
	Speaker    string     `json:"speaker"`
	Text       string     `json:"text,omitzero"`
	Confidence float64    `json:"confidence"`
}

// SpeechTurns carries ordered diarization output and optional restart state.
type SpeechTurns struct {
	Source AudioReference `json:"source"`
	State  artifact.ID    `json:"state,omitzero"`
	Turns  []SpeechTurn   `json:"turns"`
}

// Validate permits overlapping speakers but requires stable start ordering.
func (turns SpeechTurns) Validate() error {
	if err := turns.Source.Validate(); err != nil {
		return err
	}
	if turns.State.Valid() && turns.State.Kind() != artifact.KindCheckpoint {
		return errors.New("audio contract: invalid speech-turn authority")
	}
	var priorStart uint64
	for index, turn := range turns.Turns {
		if err := turn.Span.Validate(); err != nil {
			return fmt.Errorf("audio contract: speech turn %d: %w", index, err)
		}
		if !textcheck.BoundedToken(turn.Speaker, len(turn.Speaker), "") ||
			!checked.UnitInterval64(turn.Confidence) || index > 0 && turn.Span.Start < priorStart {
			return fmt.Errorf("audio contract: invalid speech turn %d", index)
		}
		priorStart = turn.Span.Start
	}
	return nil
}

// ActivitySegment is one detected active-speech interval.
type ActivitySegment struct {
	Span       SampleSpan `json:"span"`
	Confidence float64    `json:"confidence"`
}

// ActivitySegments carries non-overlapping activity output and optional state.
type ActivitySegments struct {
	Source   AudioReference    `json:"source"`
	State    artifact.ID       `json:"state,omitzero"`
	Segments []ActivitySegment `json:"segments"`
}

// Validate requires canonical non-overlapping activity order.
func (segments ActivitySegments) Validate() error {
	if err := segments.Source.Validate(); err != nil {
		return err
	}
	if segments.State.Valid() && segments.State.Kind() != artifact.KindCheckpoint {
		return errors.New("audio contract: invalid activity state")
	}
	var priorEnd uint64
	for index, segment := range segments.Segments {
		if err := segment.Span.Validate(); err != nil {
			return fmt.Errorf("audio contract: activity segment %d: %w", index, err)
		}
		if !checked.UnitInterval64(segment.Confidence) || index > 0 && segment.Span.Start < priorEnd {
			return fmt.Errorf("audio contract: invalid activity segment %d", index)
		}
		priorEnd = segment.Span.End
	}
	return nil
}

// ConvertedAudio binds transformed audio to its exact source.
type ConvertedAudio struct {
	Source AudioReference `json:"source"`
	Output AudioReference `json:"output"`
}

// Validate rejects incomplete conversion lineage.
func (converted ConvertedAudio) Validate() error {
	if err := converted.Source.Validate(); err != nil {
		return err
	}
	return validateAudioOutput(converted.Output)
}

// GeneratedAudio binds generated bytes to exact context and optional restart state.
type GeneratedAudio struct {
	Context artifact.ID    `json:"context"`
	State   artifact.ID    `json:"state,omitzero"`
	Output  AudioReference `json:"output"`
}

// Validate rejects generated audio without context or typed output geometry.
func (generated GeneratedAudio) Validate() error {
	if !generated.Context.Valid() || generated.State.Valid() && generated.State.Kind() != artifact.KindCheckpoint {
		return errors.New("audio contract: invalid generation context or state")
	}
	return validateAudioOutput(generated.Output)
}

// Clone returns caller-owned speech-turn slices.
func (turns SpeechTurns) Clone() SpeechTurns {
	turns.Turns = slices.Clone(turns.Turns)
	return turns
}

// Clone returns caller-owned activity-segment slices.
func (segments ActivitySegments) Clone() ActivitySegments {
	segments.Segments = slices.Clone(segments.Segments)
	return segments
}

func validateAudioOutput(reference AudioReference) error {
	if err := reference.Validate(); err != nil {
		return err
	}
	if reference.Audio.Kind() != artifact.KindOutput {
		return errors.New("audio contract: result audio must be an output artifact")
	}
	return nil
}
