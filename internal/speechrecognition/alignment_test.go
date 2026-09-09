package speechrecognition

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/hfbpe"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
)

func testAlignmentLease(t *testing.T, store *overgodb.Store, base recipe.Definition, wave []byte, source recipecontract.AudioReference, policy dataset.AudioInspectionPolicy, binding RunBinding) {
	t.Helper()
	profile, err := NewAlignmentProfile(AlignmentProfile{FrameMapping: "pooled-hop-cells", WordMapping: "whitespace-token-spans", BlankBoundary: "excluded", Confidence: "geometric-mean-target-probability"})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := profile.Batch("fixture/alignment/profile")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.AlignmentDefinition(base, profile.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(t.Context(), store, "fixture/alignment/recipe", definition); err != nil {
		t.Fatal(err)
	}
	condition, err := artifact.JSONContent(transcriptionContract, recipecontract.Transcription{Source: source, Text: "b", Language: "en"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "fixture/alignment/condition", Contents: []artifact.Content{condition}}); err != nil {
		t.Fatal(err)
	}
	session, err := LoadSession(t.Context(), store, definition.ID, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(context.WithoutCancel(t.Context()))
	lease, err := session.Lease(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	request := AlignmentRequest{Transcription: condition.Descriptor.ID, Span: recipecontract.SampleSpan{Start: 16, End: 112}}
	origin := dataset.AudioPayloadOrigin{Container: source.Audio}
	binding.Key = "fixture/alignment/accepted"
	result, run, err := lease.Align(t.Context(), wave, origin, policy, request, binding)
	if err != nil {
		t.Fatal(err)
	}
	if run.Recipe != definition.ID || run.Outcome != runrecord.OutcomeSucceeded || len(result.Items) != 1 || result.Items[0].Text != "b" || result.Items[0].Span.Start < request.Span.Start || result.Items[0].Span.End > request.Span.End {
		t.Fatalf("alignment binding: %+v %+v", result, run)
	}
	persisted, err := RequireAlignment(t.Context(), store, run.Outputs[0])
	if err != nil || !reflect.DeepEqual(persisted, result) {
		t.Fatalf("output: %+v %v", persisted, err)
	}
	assertRequest := func(run runrecord.Run, want AlignmentRequest) {
		t.Helper()
		found := false
		for _, id := range run.Inputs {
			content, present, err := artifact.ReadContent(t.Context(), store, id)
			if err != nil {
				t.Fatal(err)
			}
			if !present || content.Descriptor.Schema != "overgo/audio-alignment-request/v1" {
				continue
			}
			var got AlignmentRequest
			if err := json.Unmarshal(content.Data, &got); err != nil || got != want {
				t.Fatalf("request differs: %+v %v", got, err)
			}
			found = true
		}
		if !found {
			t.Fatal("run omitted exact conditioning interval")
		}
	}
	assertRequest(run, request)
	bad := request
	bad.Span.End = 129
	binding.Key = "fixture/alignment/failed"
	_, failed, err := lease.Align(t.Context(), wave, origin, policy, bad, binding)
	if err == nil || failed.Outcome != runrecord.OutcomeFailed || len(failed.Outputs) != 0 {
		t.Fatalf("failure: %+v %v", failed, err)
	}
	assertRequest(failed, bad)
	if _, err := runrecord.RequireExactRun(t.Context(), store, failed.ID); err != nil {
		t.Fatal(err)
	}
	for _, phase := range failed.Phases {
		if phase.DurationNS == 0 {
			t.Fatal("unexecuted stage reported as measurement")
		}
	}
	binding.Key = "fixture/alignment/retry"
	again, _, err := lease.Align(t.Context(), wave, origin, policy, request, binding)
	if err != nil || !reflect.DeepEqual(again, result) {
		t.Fatalf("failed run contaminated workspace: %v", err)
	}
	cancelled, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	_, before := store.Head()
	_, cancelledRun, err := lease.Align(cancelled, wave, origin, policy, request, binding)
	_, after := store.Head()
	if !errors.Is(err, context.Canceled) || cancelledRun.ID.Valid() || before != after {
		t.Fatalf("cancellation mutated store: %v %d -> %d", err, before, after)
	}
	if _, _, err := lease.Transcribe(t.Context(), wave, origin, policy, binding); err == nil {
		t.Fatal("alignment lease recognized")
	}
	if _, err := lease.PrepareCTC(t.Context(), testWave(128), 16000, "b"); err == nil {
		t.Fatal("alignment lease trained")
	}
	if _, err := session.OpenStream(t.Context(), StreamRequest{Source: source}); err == nil {
		t.Fatal("alignment session streamed")
	}
	if err := lease.Release(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := lease.Align(t.Context(), wave, origin, policy, request, binding); err == nil {
		t.Fatal("released lease aligned")
	}
	if snapshot := session.Snapshot(); snapshot.Active != 0 || snapshot.Waiting != 0 {
		t.Fatalf("residency leaked: %+v", snapshot)
	}
}

func TestAlignmentTargetMapping(t *testing.T) {
	directory, _ := transcriptionFixtureModel(t)
	tokenizer, err := hfbpe.Load(directory)
	if err != nil {
		t.Fatal(err)
	}
	words, targets, owners, err := alignmentTargets(tokenizer, "bc")
	if err != nil || !reflect.DeepEqual(words, []string{"bc"}) || !reflect.DeepEqual(targets, []int{1, 2}) || !reflect.DeepEqual(owners, []int{0, 0}) {
		t.Fatalf("mapping: %v %v %v %v", words, targets, owners, err)
	}
	for _, text := range []string{"", " b", "b ", "b  c", "b\tc", "B", "b c"} {
		t.Run(text, func(t *testing.T) {
			if _, _, _, err := alignmentTargets(tokenizer, text); err == nil {
				t.Fatal("unsupported text mapping admitted")
			}
		})
	}
}

func TestAlignmentProfileRefusesUndeclaredMapping(t *testing.T) {
	valid := AlignmentProfile{FrameMapping: "pooled-hop-cells", WordMapping: "whitespace-token-spans", BlankBoundary: "excluded", Confidence: "geometric-mean-target-probability"}
	for _, mutate := range []func(*AlignmentProfile){
		func(p *AlignmentProfile) { p.FrameMapping = "fixed-model-stride" },
		func(p *AlignmentProfile) { p.WordMapping = "language-normalization" },
		func(p *AlignmentProfile) { p.BlankBoundary = "midpoint" },
		func(p *AlignmentProfile) { p.Confidence = "word-correctness" },
	} {
		changed := valid
		mutate(&changed)
		if _, err := NewAlignmentProfile(changed); err == nil {
			t.Fatal("unsupported declaration admitted")
		}
	}
}

func TestAlignedWordSampleCells(t *testing.T) {
	path := CTCAlignment{States: []int{0, 1, 1, 2, 3, 4}}
	emissions := make([]float32, 18)
	emissions[4], emissions[7], emissions[14] = -1, -3, -2
	got, err := alignedWords(path, emissions, []int{1, 2}, []int{0, 1}, []string{"one", "two"}, 3, 4, 100, 19)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Span != (recipecontract.SampleSpan{Start: 104, End: 112}) || got[1].Span != (recipecontract.SampleSpan{Start: 116, End: 119}) || got[0].Confidence != math.Exp(-2) || got[1].Confidence != math.Exp(-2) {
		t.Fatalf("sample cells: %+v", got)
	}
	for _, test := range []struct {
		name            string
		offset, samples uint64
	}{{"padding-only", 0, 16}, {"overflow", math.MaxUint64, 19}} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := alignedWords(path, emissions, []int{1, 2}, []int{0, 1}, []string{"one", "two"}, 3, 4, test.offset, test.samples); err == nil {
				t.Fatal("invalid source extent admitted")
			}
		})
	}
	if _, err := alignedWords(CTCAlignment{States: []int{0}}, emissions, []int{1}, []int{0}, []string{"one"}, 3, 4, 0, 19); err == nil {
		t.Fatal("unsupported word admitted")
	}
}
