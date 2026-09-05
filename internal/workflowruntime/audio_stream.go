package workflowruntime

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/recipecontract"
)

// AudioStreamPolicy binds chunk bounds and decoded format to one model. Bounds
// count samples per channel, not interleaved scalar values. This profile is the
// AudioReference.Profile authority for a stream; it is not frontend FFT geometry.
type AudioStreamPolicy struct {
	Version           uint16                     `json:"version"`
	Model             artifact.ID                `json:"model"`
	Format            recipecontract.AudioFormat `json:"format"`
	MaxChunkSamples   uint64                     `json:"max_chunk_samples"`
	MaxOverlapSamples uint64                     `json:"max_overlap_samples"`
	ID                artifact.ID                `json:"-"`
}

var audioStreamPolicyCodec = artifact.JSONDocumentCodec(
	"audio stream policy", artifact.KindProfile,
	"application/vnd.overgo.audio-stream-policy+json", "overgo/audio-stream-policy/v1",
	func(p *AudioStreamPolicy) error {
		if p.Version != artifact.InitialDocumentVersion || p.Model.Kind() != artifact.KindModel ||
			p.MaxChunkSamples == 0 || p.MaxOverlapSamples >= p.MaxChunkSamples {
			return errors.New("audio stream: invalid policy")
		}
		return p.Format.Validate()
	},
	func(p AudioStreamPolicy) artifact.ID { return p.ID },
	func(p *AudioStreamPolicy, id artifact.ID) { p.ID = id },
	func(p AudioStreamPolicy) AudioStreamPolicy { return p },
)

// Batch identifies and packages a fresh policy with its exact model dependency.
func (p AudioStreamPolicy) Batch(key string) (artifact.Batch, error) {
	p, err := audioStreamPolicyCodec.NewInitial(p)
	if err != nil {
		return artifact.Batch{}, err
	}
	return audioStreamPolicyCodec.Batch(key, p, p.lineage(), nil)
}

func (p AudioStreamPolicy) lineage() []artifact.Lineage {
	return artifact.DependencyLineage(p.ID, p.Model)
}

// AudioStreamChunk identifies one bounded source interval. Overlap supplies
// context; Discontinuity explicitly drops prior model state without rewinding.
// An empty interval is legal only as an artifact-free final flush at the cursor.
type AudioStreamChunk struct {
	Sequence      uint64                    `json:"sequence"`
	Span          recipecontract.SampleSpan `json:"span"`
	Audio         artifact.ID               `json:"audio,omitzero"`
	Discontinuity bool                      `json:"discontinuity,omitzero"`
	Final         bool                      `json:"final,omitzero"`
}

// AudioStreamWork is an immutable invocation: NewSpan excludes repeated overlap.
// A processor restores PreviousState on each invocation, or resets when Reset is
// true. It must bind Model and Source.Profile before executing model-specific code.
type AudioStreamWork struct {
	Source        recipecontract.AudioReference
	Model         artifact.ID
	Format        recipecontract.AudioFormat
	Chunk         AudioStreamChunk
	NewSpan       recipecontract.SampleSpan
	PreviousState artifact.ID
	Reset         bool
}

// AudioStreamResult identifies a published output (possibly an empty event) and
// the complete model state required to restart after that output.
type AudioStreamResult struct {
	Output artifact.ID `json:"output,omitzero"`
	State  artifact.ID `json:"state,omitzero"`
}

// Validate requires typed output and restart-state identities.
func (r AudioStreamResult) Validate() error {
	if r.Output.Kind() != artifact.KindOutput || r.State.Kind() != artifact.KindCheckpoint {
		return errors.New("audio stream: incomplete result")
	}
	return nil
}

type audioStreamCheckpoint struct {
	Version      uint16                        `json:"version"`
	Source       recipecontract.AudioReference `json:"source"`
	NextSequence uint64                        `json:"next_sequence"`
	SegmentStart uint64                        `json:"segment_start"`
	EndSample    uint64                        `json:"end_sample"`
	Finalized    bool                          `json:"finalized"`
	Result       AudioStreamResult             `json:"result"`
	ID           artifact.ID                   `json:"-"`
}

var audioStreamCheckpointCodec = artifact.JSONDocumentCodec(
	"audio stream checkpoint", artifact.KindCheckpoint,
	"application/vnd.overgo.audio-stream-checkpoint+json", "overgo/audio-stream-checkpoint/v1",
	func(c *audioStreamCheckpoint) error {
		if c.Version != artifact.InitialDocumentVersion {
			return errors.New("audio stream: invalid checkpoint version")
		}
		if err := c.Source.Validate(); err != nil {
			return err
		}
		if c.NextSequence == 0 {
			if c.SegmentStart != 0 || c.EndSample != 0 || c.Finalized || c.Result != (AudioStreamResult{}) {
				return errors.New("audio stream: nonempty initial checkpoint")
			}
			return nil
		}
		if c.SegmentStart > c.EndSample || c.SegmentStart == c.EndSample && (c.EndSample != 0 || !c.Finalized) {
			return errors.New("audio stream: invalid checkpoint segment")
		}
		return c.Result.Validate()
	},
	func(c audioStreamCheckpoint) artifact.ID { return c.ID },
	func(c *audioStreamCheckpoint, id artifact.ID) { c.ID = id },
	func(c audioStreamCheckpoint) audioStreamCheckpoint { return c },
)

func (c audioStreamCheckpoint) lineage() []artifact.Lineage {
	parents := []artifact.ID{c.Source.Audio, c.Source.Profile}
	if c.NextSequence != 0 {
		parents = append(parents, c.Result.Output, c.Result.State)
	}
	return artifact.DependencyLineage(c.ID, parents...)
}

// AudioStreamCursor owns completed chunk progress, never waveform buffers. It
// is synchronous; the session owner must serialize Prepare, Complete and Batch.
// Its zero value rejects work. Restart replays from a published chunk boundary,
// not an in-flight call; consumers deduplicate outputs by artifact identity.
type AudioStreamCursor struct {
	policy AudioStreamPolicy
	state  audioStreamCheckpoint
}

// LoadAudioStream validates exact stored authority before creating or restoring
// a cursor. A checkpoint from a different source or policy cannot be resumed.
func LoadAudioStream(ctx context.Context, reader artifact.Reader, source recipecontract.AudioReference, resume artifact.ID) (*AudioStreamCursor, error) {
	if err := source.Validate(); err != nil {
		return nil, err
	}
	p, err := audioStreamPolicyCodec.RequireExactLineage(ctx, reader, source.Profile, AudioStreamPolicy.lineage)
	if err != nil {
		return nil, err
	}
	if _, found, err := reader.Artifact(ctx, source.Audio); err != nil || !found {
		return nil, errors.Join(errors.New("audio stream: source is absent"), err)
	}
	c := &AudioStreamCursor{policy: p, state: audioStreamCheckpoint{Version: artifact.InitialDocumentVersion, Source: source}}
	if resume.Valid() {
		c.state, err = audioStreamCheckpointCodec.RequireExactLineage(ctx, reader, resume, audioStreamCheckpoint.lineage)
		if err != nil {
			return nil, err
		}
		if c.state.Source != source {
			return nil, errors.New("audio stream: checkpoint source differs")
		}
	}
	return c, nil
}

// Prepare validates ordering and geometry without advancing the cursor.
func (c *AudioStreamCursor) Prepare(chunk AudioStreamChunk) (AudioStreamWork, error) {
	var work AudioStreamWork
	if c == nil || !c.policy.ID.Valid() || c.state.Finalized || chunk.Sequence != c.state.NextSequence {
		return work, errors.New("audio stream: unavailable cursor or out-of-order chunk")
	}
	if _, ok := checked.Add64(chunk.Sequence, 1); !ok {
		return work, errors.New("audio stream: sequence overflow")
	}
	span, end := chunk.Span, c.state.EndSample
	if span.Start == span.End {
		if !chunk.Final || span.Start != end || chunk.Audio.Valid() || chunk.Discontinuity {
			return work, errors.New("audio stream: invalid final flush")
		}
	} else {
		if err := span.Validate(); err != nil {
			return work, err
		}
		if span.End-span.Start > c.policy.MaxChunkSamples || span.End <= end {
			return work, errors.New("audio stream: oversized or nonadvancing chunk")
		}
		if err := (recipecontract.AudioReference{Audio: chunk.Audio, Profile: c.policy.ID}).Validate(); err != nil {
			return work, err
		}
		if chunk.Discontinuity {
			if span.Start < end {
				return work, errors.New("audio stream: discontinuity cannot rewind")
			}
		} else if span.Start < c.state.SegmentStart || span.Start > end || end-span.Start > c.policy.MaxOverlapSamples {
			return work, errors.New("audio stream: gap, cross-segment or excessive overlap")
		}
	}
	work = AudioStreamWork{
		Source: c.state.Source, Model: c.policy.Model, Format: c.policy.Format, Chunk: chunk,
		NewSpan:       recipecontract.SampleSpan{Start: max(end, span.Start), End: span.End},
		PreviousState: c.state.Result.State, Reset: c.state.NextSequence == 0 || chunk.Discontinuity,
	}
	if work.Reset {
		work.PreviousState = artifact.ID{}
	}
	return work, nil
}

// Complete advances only an unchanged prepared invocation with a typed result.
// The execution owner must first verify that result artifacts were published.
func (c *AudioStreamCursor) Complete(work AudioStreamWork, result AudioStreamResult) error {
	expected, err := c.Prepare(work.Chunk)
	if err != nil {
		return err
	}
	if expected != work {
		return errors.New("audio stream: stale or altered invocation")
	}
	if err := result.Validate(); err != nil {
		return err
	}
	c.state.NextSequence++
	c.state.ID = artifact.ID{}
	if work.Reset {
		c.state.SegmentStart = work.Chunk.Span.Start
	}
	c.state.EndSample, c.state.Finalized, c.state.Result = work.Chunk.Span.End, work.Chunk.Final, result
	return nil
}

// Batch packages the last completed boundary and its exact restart dependencies.
func (c *AudioStreamCursor) Batch(key string) (artifact.Batch, error) {
	if c == nil || !c.policy.ID.Valid() {
		return artifact.Batch{}, errors.New("audio stream: uninitialized cursor")
	}
	state, err := audioStreamCheckpointCodec.NewInitial(c.state)
	if err != nil {
		return artifact.Batch{}, err
	}
	return audioStreamCheckpointCodec.Batch(key, state, state.lineage(), nil)
}
