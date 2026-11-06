package workflowruntime

import (
	"math"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipecontract"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func audioCursorFixture(t *testing.T) (*overgodb.Store, recipecontract.AudioReference, AudioStreamResult, *AudioStreamCursor) {
	t.Helper()
	store := overgodbtest.Open(t, t.TempDir())
	model := testutil.ArtifactID(t, artifact.KindModel, "stream-model")
	source := recipecontract.AudioReference{Audio: testutil.ArtifactID(t, artifact.KindFile, "source")}
	result := AudioStreamResult{
		Output: testutil.ArtifactID(t, artifact.KindOutput, "event"),
		State:  testutil.ArtifactID(t, artifact.KindCheckpoint, "model-state"),
	}
	for _, id := range []artifact.ID{model, source.Audio, result.Output, result.State} {
		testutil.PublishArtifact(t, store, id)
	}
	batch := testcheck.Must((AudioStreamPolicy{
		Model: model, Format: recipecontract.AudioFormat{SampleRate: 16_000, Channels: 1, Encoding: "pcm-f32le"},
		MaxChunkSamples: 8, MaxOverlapSamples: 2,
	}).Batch("stream-policy"))(t)
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	source.Profile = batch.Contents[0].Descriptor.ID
	cursor := testcheck.Must(LoadAudioStream(t.Context(), store, source, artifact.ID{}))(t)
	return store, source, result, cursor
}

func TestAudioStreamOverlapDiscontinuityAndRestart(t *testing.T) {
	store, source, result, cursor := audioCursorFixture(t)
	steps := []struct {
		span, unseen                recipecontract.SampleSpan
		reset, discontinuity, final bool
	}{
		{recipecontract.SampleSpan{Start: 0, End: 8}, recipecontract.SampleSpan{Start: 0, End: 8}, true, false, false},
		{recipecontract.SampleSpan{Start: 6, End: 14}, recipecontract.SampleSpan{Start: 8, End: 14}, false, false, false},
		{recipecontract.SampleSpan{Start: 20, End: 28}, recipecontract.SampleSpan{Start: 20, End: 28}, true, true, false},
		{recipecontract.SampleSpan{Start: 28, End: 28}, recipecontract.SampleSpan{Start: 28, End: 28}, false, false, true},
	}
	for index, step := range steps {
		chunk := AudioStreamChunk{Sequence: uint64(index), Span: step.span, Audio: source.Audio, Discontinuity: step.discontinuity, Final: step.final}
		if step.final {
			chunk.Audio = artifact.ID{}
		}
		work, err := cursor.Prepare(chunk)
		if err != nil || work.NewSpan != step.unseen || work.Reset != step.reset || work.Model != cursor.policy.Model {
			t.Fatalf("step %d: work=%+v err=%v", index, work, err)
		}
		if work.Reset == work.PreviousState.Valid() {
			t.Fatalf("step %d: reset/state conflict %+v", index, work)
		}
		batch := testcheck.Must(cursor.Batch("before/" + chunk.Audio.String()))(t)
		batch.Key = batch.Contents[0].Descriptor.ID.String()
		if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
			t.Fatal(err)
		}
		restored := testcheck.Must(LoadAudioStream(t.Context(), store, source, batch.Contents[0].Descriptor.ID))(t)
		replayed, err := restored.Prepare(chunk)
		if err != nil || replayed != work {
			t.Fatalf("step %d: restart differs: %+v %v", index, replayed, err)
		}
		for _, active := range []*AudioStreamCursor{cursor, restored} {
			testcheck.NoError(t, active.Complete(work, result))
		}
		if cursor.state != restored.state {
			t.Fatal("restored completion differs")
		}
	}
	if _, err := cursor.Prepare(AudioStreamChunk{Sequence: 4}); err == nil {
		t.Fatal("finalized stream accepted another chunk")
	}
}

func TestAudioStreamRejectsInvalidTransitions(t *testing.T) {
	_, source, result, cursor := audioCursorFixture(t)
	first := AudioStreamChunk{Span: recipecontract.SampleSpan{End: 8}, Audio: source.Audio}
	work := testcheck.Must(cursor.Prepare(first))(t)
	altered := work
	altered.Reset = false
	if err := cursor.Complete(altered, result); err == nil {
		t.Fatal("altered work accepted")
	}
	if err := cursor.Complete(work, AudioStreamResult{}); err == nil {
		t.Fatal("untyped result accepted")
	}
	testcheck.NoError(t, cursor.Complete(work, result))
	before := cursor.state
	for _, tc := range []struct {
		name  string
		chunk AudioStreamChunk
	}{
		{"duplicate", first},
		{"gap", AudioStreamChunk{Sequence: 1, Span: recipecontract.SampleSpan{Start: 9, End: 12}, Audio: source.Audio}},
		{"excess-overlap", AudioStreamChunk{Sequence: 1, Span: recipecontract.SampleSpan{Start: 5, End: 12}, Audio: source.Audio}},
		{"oversize", AudioStreamChunk{Sequence: 1, Span: recipecontract.SampleSpan{Start: 8, End: 17}, Audio: source.Audio}},
		{"no-progress", AudioStreamChunk{Sequence: 1, Span: recipecontract.SampleSpan{Start: 6, End: 8}, Audio: source.Audio}},
		{"rewind-reset", AudioStreamChunk{Sequence: 1, Span: recipecontract.SampleSpan{Start: 6, End: 12}, Audio: source.Audio, Discontinuity: true}},
		{"empty-nonfinal", AudioStreamChunk{Sequence: 1, Span: recipecontract.SampleSpan{Start: 8, End: 8}}},
		{"misplaced-final", AudioStreamChunk{Sequence: 1, Span: recipecontract.SampleSpan{Start: 9, End: 9}, Final: true}},
		{"final-with-audio", AudioStreamChunk{Sequence: 1, Span: recipecontract.SampleSpan{Start: 8, End: 8}, Audio: source.Audio, Final: true}},
		{"missing-audio", AudioStreamChunk{Sequence: 1, Span: recipecontract.SampleSpan{Start: 8, End: 12}}},
		{"reversed", AudioStreamChunk{Sequence: 1, Span: recipecontract.SampleSpan{Start: 9, End: 8}, Audio: source.Audio}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := cursor.Prepare(tc.chunk); err == nil || cursor.state != before {
				t.Fatalf("invalid transition admitted or mutated state: %v", err)
			}
		})
	}
	cursor.state.NextSequence = math.MaxUint64
	if _, err := cursor.Prepare(AudioStreamChunk{Sequence: math.MaxUint64, Span: recipecontract.SampleSpan{Start: 8, End: 8}, Final: true}); err == nil {
		t.Fatal("sequence overflow admitted")
	}
}

func TestAudioStreamRejectsUnboundRestart(t *testing.T) {
	store, source, _, cursor := audioCursorFixture(t)
	batch := testcheck.Must(cursor.Batch("restart"))(t)
	batch.Lineage = nil
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadAudioStream(t.Context(), store, source, batch.Contents[0].Descriptor.ID); err == nil {
		t.Fatal("missing checkpoint lineage accepted")
	}
	batch = testcheck.Must(cursor.Batch("bound-restart"))(t)
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	source.Audio = testutil.ArtifactID(t, artifact.KindFile, "other-source")
	testutil.PublishArtifact(t, store, source.Audio)
	if _, err := LoadAudioStream(t.Context(), store, source, batch.Contents[0].Descriptor.ID); err == nil {
		t.Fatal("cross-source restart accepted")
	}
}

func TestAudioStreamEmptyFinalAndZeroValue(t *testing.T) {
	_, _, result, cursor := audioCursorFixture(t)
	work, err := cursor.Prepare(AudioStreamChunk{Final: true})
	if err != nil || !work.Reset || work.NewSpan != (recipecontract.SampleSpan{}) {
		t.Fatalf("empty final: %+v %v", work, err)
	}
	testcheck.NoError(t, cursor.Complete(work, result))
	var zero AudioStreamCursor
	if _, err := zero.Prepare(AudioStreamChunk{}); err == nil {
		t.Fatal("zero cursor admitted work")
	}
	if _, err := zero.Batch("zero"); err == nil {
		t.Fatal("zero cursor produced checkpoint")
	}
}

func TestAudioStreamPolicyAuthority(t *testing.T) {
	store, source, _, cursor := audioCursorFixture(t)
	for _, tc := range []struct {
		name   string
		change func(*AudioStreamPolicy)
	}{
		{"empty-bound", func(p *AudioStreamPolicy) { p.MaxChunkSamples = 0 }},
		{"overlap-covers-chunk", func(p *AudioStreamPolicy) { p.MaxOverlapSamples = p.MaxChunkSamples }},
		{"missing-model", func(p *AudioStreamPolicy) { p.Model = artifact.ID{} }},
		{"invalid-format", func(p *AudioStreamPolicy) { p.Format.Channels = 0 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			policy := cursor.policy
			tc.change(&policy)
			if _, err := policy.Batch(tc.name); err == nil {
				t.Fatal("invalid policy admitted")
			}
		})
	}
	checkpoint := testcheck.Must(cursor.Batch("checkpoint"))(t)
	if _, err := artifact.CommitBatch(t.Context(), store, checkpoint); err != nil {
		t.Fatal(err)
	}
	policy := cursor.policy
	policy.MaxChunkSamples++
	batch := testcheck.Must(policy.Batch("changed-policy"))(t)
	lineage := batch.Lineage
	batch.Lineage = nil
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	source.Profile = batch.Contents[0].Descriptor.ID
	if _, err := LoadAudioStream(t.Context(), store, source, artifact.ID{}); err == nil {
		t.Fatal("policy without model lineage admitted")
	}
	batch.Key, batch.Lineage = "bound-policy", lineage
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadAudioStream(t.Context(), store, source, checkpoint.Contents[0].Descriptor.ID); err == nil {
		t.Fatal("cross-policy restart admitted")
	}
	checkpoint.Key = "extra-checkpoint-lineage"
	checkpoint.Lineage = append(checkpoint.Lineage, artifact.DependencyLineage(checkpoint.Contents[0].Descriptor.ID, policy.Model)...)
	if _, err := artifact.CommitBatch(t.Context(), store, checkpoint); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadAudioStream(t.Context(), store, cursor.state.Source, checkpoint.Contents[0].Descriptor.ID); err == nil {
		t.Fatal("extra checkpoint lineage admitted")
	}
}

func TestAudioStreamOverlapCannotCrossDiscontinuity(t *testing.T) {
	store, source, result, cursor := audioCursorFixture(t)
	for _, chunk := range []AudioStreamChunk{
		{Span: recipecontract.SampleSpan{End: 8}, Audio: source.Audio},
		{Sequence: 1, Span: recipecontract.SampleSpan{Start: 20, End: 21}, Audio: source.Audio, Discontinuity: true},
	} {
		work := testcheck.Must(cursor.Prepare(chunk))(t)
		testcheck.NoError(t, cursor.Complete(work, result))
	}
	batch := testcheck.Must(cursor.Batch("discontinuity"))(t)
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	restored := testcheck.Must(LoadAudioStream(t.Context(), store, source, batch.Contents[0].Descriptor.ID))(t)
	for _, active := range []*AudioStreamCursor{cursor, restored} {
		crossing := AudioStreamChunk{Sequence: 2, Span: recipecontract.SampleSpan{Start: 19, End: 24}, Audio: source.Audio}
		if _, err := active.Prepare(crossing); err == nil {
			t.Fatal("overlap reintroduced skipped samples")
		}
		crossing.Span.Start = 20
		work, err := active.Prepare(crossing)
		if err != nil || work.NewSpan != (recipecontract.SampleSpan{Start: 21, End: 24}) {
			t.Fatalf("valid segment overlap rejected: %+v %v", work, err)
		}
	}
}
