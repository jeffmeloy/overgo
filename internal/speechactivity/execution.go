package speechactivity

import (
	"context"
	"errors"
	"math"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/checked"
	"overgo/internal/media"
	"overgo/internal/recipecontract"
	"overgo/internal/strictjson"
	"overgo/internal/workflowruntime"
)

var activityContract = artifact.JSONContract(artifact.KindOutput, "overgo/audio-activity/v1")
var stateContract = artifact.JSONContract(artifact.KindCheckpoint, "overgo/audio-activity-state/v1")

// DetectionWorkspace owns reusable numeric scratch for one synchronous caller.
// Its zero value is usable. Separate slots must never share it concurrently.
type DetectionWorkspace struct {
	frontend audiodsp.Workspace
	stream   audiodsp.StreamWorkspace
	network  Workspace
}

type detectionState struct {
	Recipe      artifact.ID                   `json:"recipe"`
	Source      recipecontract.AudioReference `json:"source"`
	Origin      uint64                        `json:"origin"`
	Frontend    audiodsp.StreamState          `json:"frontend"`
	Network     [][]float32                   `json:"network"`
	Boundary    BoundaryState                 `json:"boundary"`
	ScoreSum    float64                       `json:"score_sum"`
	ScoreFrames uint64                        `json:"score_frames"`
}

func (d *Detector) decode(ctx context.Context, id artifact.ID) (media.DecodedAudio, error) {
	content, found, err := artifact.ReadContent(ctx, d.repository, id)
	if err != nil || !found {
		return media.DecodedAudio{}, errors.Join(errors.New("speech activity: source bytes absent"), err)
	}
	if uint64(len(content.Data)) > d.memoryBytes {
		return media.DecodedAudio{}, errors.New("speech activity: encoded input exceeds component budget")
	}
	if err := content.Validate(); err != nil {
		return media.DecodedAudio{}, err
	}
	audio, _, err := media.DecodeAudio(ctx, content.Data, d.memoryBytes/4)
	if err != nil {
		return media.DecodedAudio{}, err
	}
	if audio.Format.Channels != 1 || audio.Format.SampleRate != uint64(d.profile.Frontend.SampleRate) || audio.Format.Encoding != "pcm-f32le" {
		return media.DecodedAudio{}, errors.New("speech activity: decoded audio differs from frontend format")
	}
	return audio, nil
}

// Detect executes and publishes whole-recording activity. Confidence is the
// mean raw classifier probability across each selected frame interval, not an
// annotated-quality estimate. Input bytes must already be published artifacts.
func (d *Detector) Detect(ctx context.Context, source recipecontract.AudioReference, w *DetectionWorkspace) (recipecontract.ActivitySegments, artifact.ID, error) {
	if d == nil || d.frontend == nil || ctx == nil || w == nil {
		return recipecontract.ActivitySegments{}, artifact.ID{}, errors.New("speech activity: incomplete offline execution")
	}
	if err := source.Validate(); err != nil {
		return recipecontract.ActivitySegments{}, artifact.ID{}, err
	}
	audio, err := d.decode(ctx, source.Audio)
	if err != nil {
		return recipecontract.ActivitySegments{}, artifact.ID{}, err
	}
	features, frames, err := d.frontend.Process(ctx, [][]float32{audio.Samples}, d.profile.Frontend.SampleRate, &w.frontend, audiodsp.ProcessOptions{})
	if err != nil {
		return recipecontract.ActivitySegments{}, artifact.ID{}, err
	}
	probabilities, _, err := d.network.Evaluate(ctx, features, frames, nil, &w.network, nil)
	if err != nil {
		return recipecontract.ActivitySegments{}, artifact.ID{}, err
	}
	decisions, err := OfflineDecisions(ctx, probabilities, *d.profile.Offline, d.memoryBytes)
	if err != nil {
		return recipecontract.ActivitySegments{}, artifact.ID{}, err
	}
	result := recipecontract.ActivitySegments{Source: source}
	hop := d.profile.Frontend.Geometry.HopSamples
	for start := 0; start < len(decisions); {
		if !decisions[start] {
			start++
			continue
		}
		end := start
		var sum float64
		for end < len(decisions) && decisions[end] {
			sum += float64(probabilities[end])
			end++
		}
		endSample := uint64(end) * hop
		if end == len(decisions) {
			// Native offline finalization includes the analysis window and clips
			// to the actual signal. No synthetic samples or frames are generated.
			endSample = min(uint64(len(audio.Samples)), endSample+d.profile.Frontend.Geometry.WindowSamples)
		}
		result.Segments = append(result.Segments, recipecontract.ActivitySegment{Span: recipecontract.SampleSpan{Start: uint64(start) * hop, End: endSample}, Confidence: sum / float64(end-start)})
		start = end
	}
	if err := result.Validate(); err != nil {
		return recipecontract.ActivitySegments{}, artifact.ID{}, err
	}
	content, err := artifact.JSONContent(activityContract, result)
	if err != nil {
		return recipecontract.ActivitySegments{}, artifact.ID{}, err
	}
	batch, err := artifact.NewDocumentBatch("audio-activity/"+content.Descriptor.ID.String(), []artifact.Content{content}, artifact.DependencyLineage(content.Descriptor.ID, d.definition.ID, source.Audio, source.Profile), nil)
	if err == nil {
		_, err = artifact.CommitBatch(ctx, d.repository, batch)
	}
	if err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return recipecontract.ActivitySegments{}, artifact.ID{}, err
	}
	return result, content.Descriptor.ID, nil
}

func (d *Detector) restore(ctx context.Context, work workflowruntime.AudioStreamWork) (detectionState, error) {
	state := detectionState{Recipe: d.definition.ID, Source: work.Source, Origin: work.NewSpan.Start}
	if work.Reset {
		if work.PreviousState.Valid() {
			return detectionState{}, errors.New("speech activity: reset has restart authority")
		}
		return state, nil
	}
	content, found, err := artifact.ReadContent(ctx, d.repository, work.PreviousState)
	if err != nil || !found {
		return detectionState{}, errors.Join(errors.New("speech activity: restart content absent"), err)
	}
	if uint64(len(content.Data)) > d.memoryBytes {
		return detectionState{}, errors.New("speech activity: serialized state exceeds component budget")
	}
	if err := stateContract.ValidateContent(content, work.PreviousState); err != nil {
		return detectionState{}, err
	}
	if err := strictjson.DecodeBytes(content.Data, &state); err != nil {
		return detectionState{}, err
	}
	if state.Recipe != d.definition.ID || state.Source != work.Source || state.Origin > work.NewSpan.Start || state.Frontend.Samples != work.NewSpan.Start-state.Origin ||
		state.Boundary.Frame != state.Frontend.Frames || !checked.NonNegativeFinite64(state.ScoreSum) || state.ScoreFrames > state.Boundary.Frame || state.ScoreSum > float64(state.ScoreFrames) ||
		state.Frontend.Final || state.Boundary.Final || state.Frontend.Frames > 0 && len(state.Network) != len(d.network.blocks) {
		return detectionState{}, errors.New("speech activity: restart authorities or frame positions differ")
	}
	for i, cache := range state.Network {
		if i >= len(d.network.blocks) || uint64(len(cache)) != min(state.Frontend.Frames, uint64(d.network.blocks[i].context))*uint64(d.network.blocks[i].project.out) {
			return detectionState{}, errors.New("speech activity: restart cache extent differs")
		}
		for _, value := range cache {
			if !checked.Finite32(value) {
				return detectionState{}, errors.New("speech activity: non-finite restart cache")
			}
		}
	}
	return state, nil
}

// ProcessStream is the synchronous processor for capabilityruntime.OpenAudioStream.
// It restores all state from Work.PreviousState, excludes repeated overlap and
// atomically publishes output/state. Discontinuities reset framing and decisions.
// Confidence averages observed probabilities after start confirmation, excluding
// retrospective start padding. It is not a calibrated quality measurement.
func (d *Detector) ProcessStream(ctx context.Context, work workflowruntime.AudioStreamWork, w *DetectionWorkspace) (workflowruntime.AudioStreamResult, error) {
	if d == nil || d.stream == nil || d.boundary == nil || ctx == nil || w == nil || work.Model != d.profile.Model ||
		work.Format.SampleRate != uint64(d.profile.Frontend.SampleRate) || work.Format.Channels != 1 || work.Format.Encoding != "pcm-f32le" ||
		work.Chunk.Span.Start > work.NewSpan.Start || work.NewSpan.Start > work.NewSpan.End || work.NewSpan.End != work.Chunk.Span.End {
		return workflowruntime.AudioStreamResult{}, errors.New("speech activity: incompatible stream invocation")
	}
	if err := work.Source.Validate(); err != nil {
		return workflowruntime.AudioStreamResult{}, err
	}
	state, err := d.restore(ctx, work)
	if err != nil {
		return workflowruntime.AudioStreamResult{}, err
	}
	var samples []float32
	if work.Chunk.Span.End > work.Chunk.Span.Start {
		audio, err := d.decode(ctx, work.Chunk.Audio)
		if err != nil {
			return workflowruntime.AudioStreamResult{}, err
		}
		if uint64(len(audio.Samples)) != work.Chunk.Span.End-work.Chunk.Span.Start {
			return workflowruntime.AudioStreamResult{}, errors.New("speech activity: chunk interval differs from decoded samples")
		}
		samples = audio.Samples[work.NewSpan.Start-work.Chunk.Span.Start:]
	} else if !work.Chunk.Final || work.Chunk.Audio.Valid() {
		return workflowruntime.AudioStreamResult{}, errors.New("speech activity: invalid empty flush")
	}
	features, frames, next, err := d.stream.Process(ctx, samples, d.profile.Frontend.SampleRate, state.Frontend, work.Chunk.Final, &w.stream)
	if err != nil {
		return workflowruntime.AudioStreamResult{}, err
	}
	state.Frontend = next
	var probabilities []float32
	if frames > 0 {
		probabilities, state.Network, err = d.network.Evaluate(ctx, features, frames, state.Network, &w.network, nil)
		if err != nil {
			return workflowruntime.AudioStreamResult{}, err
		}
	}
	result := recipecontract.ActivitySegments{Source: work.Source}
	observed := state.Boundary.Frame
	err = d.boundary.Process(ctx, probabilities, &state.Boundary, work.Chunk.Final, func(event Boundary) error {
		if event.Started {
			state.ScoreSum, state.ScoreFrames = 0, 0
		}
		if event.Frame > observed && (state.Boundary.Start != 0 || event.Ended) {
			state.ScoreSum += event.Probability
			state.ScoreFrames++
		}
		observed = event.Frame
		if event.Ended && event.End > event.Start && event.Start > 0 {
			hop := d.profile.Frontend.Geometry.HopSamples
			if event.End-1 > (math.MaxUint64-state.Origin)/hop {
				return errors.New("speech activity: sample boundary overflows")
			}
			if state.ScoreFrames == 0 {
				return errors.New("speech activity: segment has no probability observations")
			}
			span := recipecontract.SampleSpan{Start: state.Origin + (event.Start-1)*hop, End: state.Origin + (event.End-1)*hop}
			result.Segments = append(result.Segments, recipecontract.ActivitySegment{Span: span, Confidence: state.ScoreSum / float64(state.ScoreFrames)})
		}
		if event.Ended {
			state.ScoreSum, state.ScoreFrames = 0, 0
		}
		return nil
	})
	if err != nil {
		return workflowruntime.AudioStreamResult{}, err
	}
	checkpoint, err := artifact.JSONContent(stateContract, state)
	if err != nil {
		return workflowruntime.AudioStreamResult{}, err
	}
	result.State = checkpoint.Descriptor.ID
	if err := result.Validate(); err != nil {
		return workflowruntime.AudioStreamResult{}, err
	}
	output, err := artifact.JSONContent(activityContract, result)
	if err != nil {
		return workflowruntime.AudioStreamResult{}, err
	}
	parents := []artifact.ID{d.definition.ID, work.Source.Audio, work.Source.Profile}
	if work.Chunk.Audio.Valid() {
		parents = append(parents, work.Chunk.Audio)
	}
	if work.PreviousState.Valid() {
		parents = append(parents, work.PreviousState)
	}
	lineage := append(artifact.DependencyLineage(checkpoint.Descriptor.ID, parents...), artifact.DependencyLineage(output.Descriptor.ID, checkpoint.Descriptor.ID)...)
	batch, err := artifact.NewDocumentBatch("audio-activity-stream/"+checkpoint.Descriptor.ID.String(), []artifact.Content{checkpoint, output}, lineage, nil)
	if err == nil {
		_, err = artifact.CommitBatch(ctx, d.repository, batch)
	}
	if err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return workflowruntime.AudioStreamResult{}, err
	}
	return workflowruntime.AudioStreamResult{Output: output.Descriptor.ID, State: checkpoint.Descriptor.ID}, nil
}

// RequireActivity reads canonical, typed activity output and validates ordering.
func RequireActivity(ctx context.Context, reader artifact.Reader, id artifact.ID) (recipecontract.ActivitySegments, error) {
	var result recipecontract.ActivitySegments
	content, found, err := artifact.ReadContent(ctx, reader, id)
	if err != nil || !found {
		return result, errors.Join(errors.New("speech activity: output absent"), err)
	}
	if err := activityContract.ValidateContent(content, id); err != nil {
		return result, err
	}
	if err := strictjson.DecodeBytes(content.Data, &result); err != nil {
		return result, err
	}
	if err := result.Validate(); err != nil {
		return recipecontract.ActivitySegments{}, err
	}
	canonical, err := artifact.JSONContent(activityContract, result)
	if err != nil || canonical.Descriptor.ID != id {
		return recipecontract.ActivitySegments{}, errors.Join(errors.New("speech activity: output is not canonical"), err)
	}
	return result, nil
}
