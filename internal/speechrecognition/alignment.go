package speechrecognition

import (
	"context"
	"errors"
	"math"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/dataset"
	"overgo/internal/hfbpe"
	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
)

var alignmentContract = artifact.JSONContract(artifact.KindOutput, "overgo/audio-alignment/v1")
var alignmentRequestContract = artifact.JSONContract(artifact.KindEvidence, "overgo/audio-alignment-request/v1")

// AlignmentRequest names the conditioning transcript and the exact half-open
// source interval to align. Text is neither recognized nor normalized here.
type AlignmentRequest struct {
	Transcription artifact.ID               `json:"transcription"`
	Span          recipecontract.SampleSpan `json:"span"`
}

// Align executes forced alignment through this recipe's existing component
// lease. It records the conditioning transcript, source interval, inspection,
// execution recipe and code/environment binding. It is not ASR prediction.
func (lease *SpeechLease) Align(ctx context.Context, data []byte, origin dataset.AudioPayloadOrigin, policy dataset.AudioInspectionPolicy, request AlignmentRequest, binding RunBinding) (recipecontract.TimestampedAlignment, runrecord.Run, error) {
	if lease == nil || ctx == nil {
		return recipecontract.TimestampedAlignment{}, runrecord.Run{}, errors.New("alignment: incomplete lease invocation")
	}
	lease.mu.Lock()
	defer lease.mu.Unlock()
	if lease.released || lease.definition.Task != recipe.TaskAlignment || len(lease.components) != 1 {
		return recipecontract.TimestampedAlignment{}, runrecord.Run{}, errors.New("alignment: an admitted alignment lease is required")
	}
	component := lease.components[0].Model()
	if component == nil || component.transcriber == nil || component.transcriber.transducer != nil {
		return recipecontract.TimestampedAlignment{}, runrecord.Run{}, errors.New("alignment: standalone CTC component required")
	}
	bound := *component.transcriber
	bound.recipe = lease.definition
	return bound.align(ctx, data, origin, policy, request, binding, component)
}

func (transcriber *transcriptionModel) align(ctx context.Context, data []byte, origin dataset.AudioPayloadOrigin, policy dataset.AudioInspectionPolicy, request AlignmentRequest, binding RunBinding, component *transcriptionComponent) (recipecontract.TimestampedAlignment, runrecord.Run, error) {
	if binding.Key == "" || binding.Dataset.Valid() != binding.Split.Valid() ||
		binding.Dataset.Valid() && (binding.Dataset.Kind() != artifact.KindDataset || binding.Split.Kind() != artifact.KindDatasetShard) {
		return recipecontract.TimestampedAlignment{}, runrecord.Run{}, errors.New("alignment: invalid run binding")
	}
	if err := request.Span.Validate(); err != nil {
		return recipecontract.TimestampedAlignment{}, runrecord.Run{}, err
	}
	transcript, err := RequireTranscription(ctx, transcriber.repository, request.Transcription)
	if err != nil {
		return recipecontract.TimestampedAlignment{}, runrecord.Run{}, err
	}
	var walls processmeasure.Walls
	started := processmeasure.NewStopwatch()
	inspection, err := dataset.InspectAudio(ctx, transcriber.repository, data, origin, policy)
	if err != nil {
		return recipecontract.TimestampedAlignment{}, runrecord.Run{}, err
	}
	phases := completedTranscriptionPhases(walls.Elapsed(started), 0, 0, 0)
	inputs := uniqueIDs(transcriber.model, request.Transcription, binding.Dataset, binding.Split,
		inspection.Signal.Source.Audio, inspection.Signal.Source.Profile, inspection.SignalID, inspection.PolicyID, inspection.DecisionID)
	fail := func(code string, cause error) (recipecontract.TimestampedAlignment, runrecord.Run, error) {
		if ctx.Err() != nil {
			return recipecontract.TimestampedAlignment{}, runrecord.Run{}, cause
		}
		measured := walls.Elapsed(started)
		if err := walls.Err; err != nil {
			return recipecontract.TimestampedAlignment{}, runrecord.Run{}, errors.Join(cause, err)
		}
		run, err := persistSpeechRun(ctx, transcriber.repository, transcriber.recipe.ID, binding, runrecord.OutcomeFailed, inputs, nil, code, measured, phases)
		return recipecontract.TimestampedAlignment{}, run, errors.Join(cause, err)
	}
	requestContent, err := artifact.JSONContent(alignmentRequestContract, request)
	if err != nil {
		return fail("alignment-request-invalid", err)
	}
	if _, err := artifact.CommitBatch(ctx, transcriber.repository, artifact.Batch{Key: "alignment/request/" + requestContent.Descriptor.ID.String(),
		Contents: []artifact.Content{requestContent}, Lineage: artifact.DependencyLineage(requestContent.Descriptor.ID, request.Transcription)}); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return recipecontract.TimestampedAlignment{}, runrecord.Run{}, err
	}
	inputs = uniqueIDs(append(inputs, requestContent.Descriptor.ID)...)
	if inspection.Decision.Outcome != recipecontract.AudioAdmissionAccepted {
		return fail(AudioAdmissionFailure, ErrAudioAdmissionRefused)
	}
	if inspection.Signal.Format != transcriber.contract.Format || transcript.Source != inspection.Signal.Source || request.Span.End > uint64(len(inspection.Samples)) {
		return fail("alignment-source-mismatch", errors.New("alignment: transcript, format or interval differs from admitted audio"))
	}
	items, err := transcriber.alignWords(ctx, inspection.Samples[request.Span.Start:request.Span.End], int(inspection.Signal.Format.SampleRate), transcript.Text, request.Span.Start, component, &walls, phases)
	if err != nil {
		return fail("alignment-execution-failed", err)
	}
	result := recipecontract.TimestampedAlignment{Source: transcript.Source, Transcription: request.Transcription, Items: items}
	if err := result.Validate(); err != nil {
		return fail("alignment-output-invalid", err)
	}
	output, err := artifact.JSONContent(alignmentContract, result)
	if err != nil {
		return fail("alignment-output-invalid", err)
	}
	measured := walls.Elapsed(started)
	if err := walls.Err; err != nil {
		return recipecontract.TimestampedAlignment{}, runrecord.Run{}, err
	}
	run, err := runrecord.NewBoundRun(transcriber.recipe.ID, runrecord.OutcomeSucceeded, inputs, []artifact.ID{output.Descriptor.ID}, "", binding.CodeCommit, binding.Environment, measured, phases)
	if err != nil {
		return recipecontract.TimestampedAlignment{}, runrecord.Run{}, err
	}
	runContent, err := run.Content()
	if err != nil {
		return recipecontract.TimestampedAlignment{}, runrecord.Run{}, err
	}
	lineage := append(run.Lineage(), artifact.DependencyLineage(output.Descriptor.ID, transcriber.recipe.ID, requestContent.Descriptor.ID, transcript.Source.Audio, transcript.Source.Profile)...)
	batch, err := artifact.NewDocumentBatch(binding.Key, []artifact.Content{output, runContent}, lineage, nil)
	if err != nil {
		return recipecontract.TimestampedAlignment{}, runrecord.Run{}, err
	}
	if _, err := artifact.CommitBatch(ctx, transcriber.repository, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return recipecontract.TimestampedAlignment{}, runrecord.Run{}, err
	}
	return result, run, nil
}

func alignmentTargets(tokenizer *hfbpe.Tokenizer, text string) (words []string, targets, owners []int, err error) {
	words = strings.Fields(text)
	if len(words) == 0 || strings.Join(words, " ") != text {
		return nil, nil, nil, errors.New("alignment: nonempty, single-space-separated target required")
	}
	whole, err := tokenizer.Encode(text)
	if err != nil {
		return nil, nil, nil, err
	}
	for index, word := range words {
		piece := word
		if index > 0 {
			piece = " " + piece
		}
		ids, err := tokenizer.Encode(piece)
		if err != nil || len(ids) == 0 {
			return nil, nil, nil, errors.Join(errors.New("alignment: word cannot be tokenized"), err)
		}
		targets = append(targets, ids...)
		for range ids {
			owners = append(owners, index)
		}
	}
	decoded, err := tokenizer.DecodeStrict(targets)
	if err != nil || decoded != text || !slices.Equal(targets, whole) {
		return nil, nil, nil, errors.New("alignment: tokenizer crosses a word boundary or changes target text")
	}
	return words, targets, owners, nil
}

func (transcriber *transcriptionModel) alignWords(ctx context.Context, samples []float32, rate int, text string, offset uint64, component *transcriptionComponent, walls *processmeasure.Walls, phases []runrecord.PhaseMetric) ([]recipecontract.AlignedText, error) {
	prepareStart := processmeasure.NewStopwatch()
	words, targets, owners, err := alignmentTargets(transcriber.tokenizer, text)
	if err != nil {
		return nil, err
	}
	frames, err := transcriber.frontend.GroupedFrames(uint64(len(samples)), transcriber.profile.Grouping)
	if err != nil {
		return nil, err
	}
	outputFrames := transcriber.encoder.outputFrames(frames)
	_, _, budget, err := alignmentStorage(outputFrames, len(targets))
	if err != nil {
		return nil, err
	}
	// Drop a previous larger path before executing the encoder under this
	// request's smaller reservation; trimming only inside Align is too late.
	component.alignment.discardExcess(budget)
	// The encoder's existing admission includes the CTC dynamic-programming
	// workspace. Tokenization, output objects and allocator overhead are not RSS
	// measurements and are not represented by this scratch reservation.
	component.text.Encoder.reservedBytes = budget
	features, frames, _, err := transcriber.frontend.ProcessGrouped(ctx, samples, rate, &component.text.Frontend, transcriber.profile.Grouping)
	phases[1].DurationNS = walls.Elapsed(prepareStart)
	if err != nil {
		return nil, err
	}
	inferStart := processmeasure.NewStopwatch()
	hidden, frames, err := transcriber.encoder.Encode(ctx, features, frames, &component.text.Encoder, nil)
	if err != nil {
		return nil, err
	}
	logits, err := transcriber.encoder.Project(ctx, hidden, frames, &component.text.Encoder)
	phases[2].DurationNS = walls.Elapsed(inferStart)
	if err != nil {
		return nil, err
	}
	postStart := processmeasure.NewStopwatch()
	defer func() { phases[3].DurationNS = walls.Elapsed(postStart) }()
	vocabulary := transcriber.encoder.VocabularySize()
	for frame := range frames {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		row := logits[frame*vocabulary : (frame+1)*vocabulary]
		maximum := float64(slices.Max(row))
		var sum float64
		for _, value := range row {
			sum += math.Exp(float64(value) - maximum)
		}
		normalizer := math.Log(sum)
		for index, value := range row {
			row[index] = float32((float64(value) - maximum) - normalizer)
		}
	}
	path, err := component.alignment.Align(ctx, logits, targets, frames, vocabulary, transcriber.profile.BlankToken, budget)
	if err != nil {
		return nil, err
	}
	hop, ok := checked.Mul64(transcriber.profile.Frontend.Geometry.HopSamples, uint64(transcriber.profile.Grouping.StackFrames))
	for _, block := range transcriber.encoder.blocks {
		if !ok {
			break
		}
		hop, ok = checked.Mul64(hop, uint64(block.convolution.stride))
	}
	if !ok || hop == 0 {
		return nil, errors.New("alignment: frame/sample mapping overflows")
	}
	return alignedWords(path, logits, targets, owners, words, vocabulary, hop, offset, uint64(len(samples)))
}

func alignedWords(path CTCAlignment, emissions []float32, targets, owners []int, words []string, vocabulary int, hop, offset, samples uint64) ([]recipecontract.AlignedText, error) {
	items := make([]recipecontract.AlignedText, len(words))
	counts := make([]int, len(words))
	for frame, state := range path.States {
		if state%2 == 0 {
			continue
		}
		target := state / 2
		word := owners[target]
		start, ok := checked.Mul64(uint64(frame), hop)
		end, endOK := checked.Add64(start, hop)
		if !ok || !endOK || start >= samples {
			return nil, errors.New("alignment: token is outside source samples")
		}
		end = min(end, samples)
		start, ok = checked.Add64(start, offset)
		end, endOK = checked.Add64(end, offset)
		if !ok || !endOK {
			return nil, errors.New("alignment: source interval overflows")
		}
		if counts[word] == 0 {
			items[word].Span.Start = start
			items[word].Text = words[word]
		}
		items[word].Span.End = end
		items[word].Confidence += float64(emissions[frame*vocabulary+targets[target]])
		counts[word]++
	}
	for index := range items {
		if counts[index] == 0 {
			return nil, errors.New("alignment: target word has no frame support")
		}
		items[index].Confidence = math.Exp(items[index].Confidence / float64(counts[index]))
	}
	return items, nil
}

// RequireAlignment loads a canonical persisted source-bound alignment output.
func RequireAlignment(ctx context.Context, reader artifact.Reader, id artifact.ID) (recipecontract.TimestampedAlignment, error) {
	var result recipecontract.TimestampedAlignment
	content, found, err := artifact.ReadContent(ctx, reader, id)
	if err != nil {
		return result, err
	}
	if !found {
		return result, errors.New("alignment: output is absent")
	}
	if err := alignmentContract.ValidateContent(content, id); err != nil {
		return result, err
	}
	if err := strictjson.DecodeBytes(content.Data, &result); err != nil {
		return result, err
	}
	if err := result.Validate(); err != nil {
		return result, err
	}
	canonical, err := artifact.JSONContent(alignmentContract, result)
	if err != nil || canonical.Descriptor.ID != id {
		return recipecontract.TimestampedAlignment{}, errors.New("alignment: output is not canonical")
	}
	return result, nil
}
