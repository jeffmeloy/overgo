package speechrecognition

import (
	"context"
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/hfbpe"
	"overgo/internal/media"
	"overgo/internal/recipecontract"
	"overgo/internal/strictjson"
	"overgo/internal/workflowruntime"
)

var transcriptionStateContract = artifact.JSONContract(artifact.KindCheckpoint, "overgo/transcription-stream-state/v1")
var transcriptionChunkContract = artifact.JSONContract(artifact.KindOutput, "overgo/transcription-chunk/v1")

// TranscriptionChunk is an ordered text suffix and its emitted token/frame
// advances. Span locates newly consumed input, not word timestamps. Empty text
// is a valid completed boundary; State identifies all restart authority.
type TranscriptionChunk struct {
	Source   recipecontract.AudioReference `json:"source"`
	Recipe   artifact.ID                   `json:"recipe"`
	Sequence uint64                        `json:"sequence"`
	Span     recipecontract.SampleSpan     `json:"span"`
	Text     string                        `json:"text"`
	Tokens   []int                         `json:"tokens,omitempty"`
	Advances []int                         `json:"advances,omitempty"`
	Final    bool                          `json:"final"`
	State    artifact.ID                   `json:"state"`
}

// RequireTranscriptionChunk reads one content-verified native streaming output.
func RequireTranscriptionChunk(ctx context.Context, reader artifact.Reader, id artifact.ID) (TranscriptionChunk, error) {
	var result TranscriptionChunk
	content, err := artifact.RequireTypedContent(ctx, reader, id)
	if err != nil {
		return result, err
	}
	if err := transcriptionChunkContract.ValidateContent(content, id); err != nil {
		return result, err
	}
	if err := strictjson.DecodeBytes(content.Data, &result); err != nil {
		return TranscriptionChunk{}, err
	}
	if err := result.Source.Validate(); err != nil {
		return TranscriptionChunk{}, err
	}
	if result.Recipe.Kind() != artifact.KindRecipe || result.State.Kind() != artifact.KindCheckpoint ||
		result.Span.Start > result.Span.End || len(result.Tokens) != len(result.Advances) {
		return TranscriptionChunk{}, errors.New("transcription stream: invalid output authority or geometry")
	}
	for i, token := range result.Tokens {
		if token < 0 || result.Advances[i] < 0 || result.Advances[i] > 1 {
			return TranscriptionChunk{}, errors.New("transcription stream: invalid output emissions")
		}
	}
	return result, nil
}

type transcriptionStreamState struct {
	Recipe   artifact.ID                   `json:"recipe"`
	Source   recipecontract.AudioReference `json:"source"`
	Origin   uint64                        `json:"origin"`
	Frontend audiodsp.StreamState          `json:"frontend"`
	Pending  []float32                     `json:"pending,omitempty"`
	Network  *transducerCheckpoint         `json:"network,omitzero"`
	Text     hfbpe.DecodeState             `json:"text"`
}

type transcriptionStreamWorkspace struct {
	frontend audiodsp.StreamWorkspace
	network  TransducerWorkspace
	chunk    []float32
}

func (transcriber *transcriptionModel) restoreStream(ctx context.Context, work workflowruntime.AudioStreamWork, w *transcriptionStreamWorkspace) (transcriptionStreamState, error) {
	state := transcriptionStreamState{Recipe: transcriber.recipe.ID, Source: work.Source, Origin: work.NewSpan.Start}
	w.network = TransducerWorkspace{}
	if work.Reset {
		if work.PreviousState.Valid() {
			return state, errors.New("transcription stream: reset has prior authority")
		}
		return state, nil
	}
	descriptor, found, err := transcriber.repository.Artifact(ctx, work.PreviousState)
	if err != nil || !found || descriptor.Size > transcriber.memory {
		return state, errors.Join(errors.New("transcription stream: restart absent or exceeds budget"), err)
	}
	content, err := artifact.RequireTypedContent(ctx, transcriber.repository, work.PreviousState)
	if err != nil {
		return state, err
	}
	if err := transcriptionStateContract.ValidateContent(content, work.PreviousState); err != nil {
		return state, err
	}
	if err := strictjson.DecodeBytes(content.Data, &state); err != nil {
		return state, err
	}
	if state.Recipe != transcriber.recipe.ID || state.Source != work.Source || state.Origin > work.NewSpan.Start ||
		state.Frontend.Samples != work.NewSpan.Start-state.Origin || state.Frontend.Final {
		return state, errors.New("transcription stream: restart recipe, source or sample position differs")
	}
	if _, _, err := transcriber.tokenizer.DecodeTextChunk(nil, state.Text, false); err != nil {
		return state, err
	}
	if state.Network == nil && (state.Text.Started || len(state.Text.Pending) != 0) {
		return state, errors.New("transcription stream: decoded text precedes acoustic execution")
	}
	first, following, _, err := transcriber.transducer.streamGeometry(nil)
	if err != nil {
		return state, err
	}
	consumed := uint64(0)
	expected := first
	if state.Network != nil {
		if err := transcriber.transducer.restore(*state.Network, &w.network); err != nil {
			return state, err
		}
		chunks := uint64(state.Network.Frames / transcriber.encoder.blocks[0].attention.block)
		var ok bool
		consumed, ok = checked.Mul64(chunks-1, uint64(following))
		if !ok {
			return state, errors.New("transcription stream: restored mel position overflows")
		}
		consumed, ok = checked.Add64(consumed, uint64(first))
		if !ok {
			return state, errors.New("transcription stream: restored mel position overflows")
		}
		expected = following
	}
	bands := transcriber.transducer.binding.Bands
	if len(state.Pending)%bands != 0 || len(state.Pending)/bands >= expected || consumed > state.Frontend.Frames ||
		state.Frontend.Frames-consumed != uint64(len(state.Pending)/bands) {
		return state, errors.New("transcription stream: restored pending mel extent differs")
	}
	for _, value := range state.Pending {
		if !checked.Finite32(value) {
			return state, errors.New("transcription stream: non-finite pending features")
		}
	}
	return state, nil
}

func (transcriber *transcriptionModel) processStream(ctx context.Context, work workflowruntime.AudioStreamWork, w *transcriptionStreamWorkspace) (workflowruntime.AudioStreamResult, error) {
	var result workflowruntime.AudioStreamResult
	if ctx == nil || w == nil || transcriber.transducer == nil || transcriber.stream == nil || work.Model != transcriber.model ||
		work.Format != transcriber.contract.Format || work.Chunk.Span.Start > work.NewSpan.Start ||
		work.NewSpan.Start > work.NewSpan.End || work.NewSpan.End != work.Chunk.Span.End {
		return result, errors.New("transcription stream: incompatible invocation")
	}
	if err := work.Source.Validate(); err != nil {
		return result, err
	}
	state, err := transcriber.restoreStream(ctx, work, w)
	if err != nil {
		return result, err
	}
	var samples []float32
	if work.Chunk.Span.End > work.Chunk.Span.Start {
		descriptor, found, err := transcriber.repository.Artifact(ctx, work.Chunk.Audio)
		if err != nil || !found || descriptor.Size > transcriber.memory {
			return result, errors.Join(errors.New("transcription stream: input absent or exceeds budget"), err)
		}
		content, found, err := artifact.ReadContent(ctx, transcriber.repository, work.Chunk.Audio)
		if err != nil || !found {
			return result, errors.Join(errors.New("transcription stream: chunk content absent"), err)
		}
		audio, _, err := media.DecodeAudio(ctx, content.Data, transcriber.memory/binaryschema.Uint32Bytes)
		if err != nil {
			return result, err
		}
		if audio.Format != work.Format || uint64(len(audio.Samples)) != work.Chunk.Span.End-work.Chunk.Span.Start {
			return result, errors.New("transcription stream: decoded format or source interval differs")
		}
		samples = audio.Samples[work.NewSpan.Start-work.Chunk.Span.Start:]
	} else if !work.Chunk.Final || work.Chunk.Audio.Valid() {
		return result, errors.New("transcription stream: invalid empty finalization")
	}
	features, _, next, err := transcriber.stream.Process(ctx, samples, transcriber.profile.Frontend.SampleRate, state.Frontend, work.Chunk.Final, &w.frontend)
	if err != nil {
		return result, err
	}
	state.Frontend = next
	output := TranscriptionChunk{Source: work.Source, Recipe: transcriber.recipe.ID, Sequence: work.Chunk.Sequence, Span: work.NewSpan, Final: work.Chunk.Final}
	if err := transcriber.consumeMel(ctx, features, &state, &output, w); err != nil {
		return result, err
	}
	checkpoint, err := artifact.JSONContent(transcriptionStateContract, state)
	if err != nil {
		return result, err
	}
	output.State = checkpoint.Descriptor.ID
	content, err := artifact.JSONContent(transcriptionChunkContract, output)
	if err != nil {
		return result, err
	}
	parents := []artifact.ID{transcriber.recipe.ID, work.Source.Audio, work.Source.Profile}
	for _, id := range []artifact.ID{work.Chunk.Audio, work.PreviousState} {
		if id.Valid() {
			parents = append(parents, id)
		}
	}
	lineage := append(artifact.DependencyLineage(checkpoint.Descriptor.ID, parents...), artifact.DependencyLineage(content.Descriptor.ID, checkpoint.Descriptor.ID)...)
	batch, err := artifact.NewDocumentBatch("transcription-stream/"+checkpoint.Descriptor.ID.String(), []artifact.Content{checkpoint, content}, lineage, nil)
	if err == nil {
		_, err = artifact.CommitBatch(ctx, transcriber.repository, batch)
	}
	if err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return result, err
	}
	return workflowruntime.AudioStreamResult{Output: content.Descriptor.ID, State: checkpoint.Descriptor.ID}, nil
}

func (transcriber *transcriptionModel) consumeMel(ctx context.Context, features []float32, state *transcriptionStreamState, output *TranscriptionChunk, w *transcriptionStreamWorkspace) error {
	first, following, _, err := transcriber.transducer.streamGeometry(nil)
	if err != nil {
		return err
	}
	bands := transcriber.transducer.binding.Bands
	capacity, ok := checked.MulInt(max(first, following), bands)
	if !ok || uint64(capacity) > transcriber.memory/binaryschema.Uint32Bytes {
		return errors.New("transcription stream: mel chunk exceeds budget")
	}
	if cap(w.chunk) < capacity {
		w.chunk = make([]float32, capacity)
	}
	var text strings.Builder
	pending := state.Pending
	for len(pending) != 0 || len(features) != 0 {
		frames := first
		if w.network.stream != nil {
			frames = following
		}
		count := frames * bands
		available, ok := checked.AddInt(len(pending), len(features))
		if !ok {
			return errors.New("transcription stream: feature extent overflows")
		}
		if available < count && !output.Final {
			break
		}
		chunk := w.chunk[:count]
		copied := copy(chunk, pending)
		used := copy(chunk[copied:], features)
		clear(chunk[copied+used:])
		features, pending = features[used:], nil
		final := output.Final && len(features) == 0
		decoded, err := transcriber.transducer.RecognizeChunk(ctx, chunk, frames, final, &w.network, nil)
		if err != nil {
			return err
		}
		part, next, err := transcriber.tokenizer.DecodeTextChunk(decoded.Tokens, state.Text, final)
		if err != nil {
			return err
		}
		text.WriteString(part)
		state.Text = next
		output.Tokens = append(output.Tokens, decoded.Tokens...)
		output.Advances = append(output.Advances, decoded.Durations...)
	}
	state.Pending = append(slices.Clone(pending), features...)
	if output.Final {
		part, next, err := transcriber.tokenizer.DecodeTextChunk(nil, state.Text, true)
		if err != nil {
			return err
		}
		text.WriteString(part)
		state.Text = next
		if w.network.stream != nil {
			w.network.stream.final = true
		}
	}
	if w.network.stream != nil {
		checkpoint, err := transcriber.transducer.checkpoint(&w.network)
		if err != nil {
			return err
		}
		state.Network = &checkpoint
	}
	output.Text = text.String()
	return nil
}
