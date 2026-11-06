package modelintake

import (
	"bytes"
	"context"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// recordingGenerator answers every prompt with the prompt's words reversed,
// reporting the prompt token count the way a runner does.
type recordingGenerator struct{}

func (recordingGenerator) Generate(_ context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	words := strings.Fields(prompt)
	if options.OnPromptEvaluated != nil {
		options.OnPromptEvaluated(inference.PromptEvaluation{Tokens: len(words)})
	}
	var ids []tokenizer.TokenID
	for index := range len(words) {
		ids = append(ids, tokenizer.TokenID(index))
	}
	for index := range min(options.MaxNewTokens, len(words)) {
		piece := words[len(words)-index-1] + " "
		if options.OnToken != nil {
			if err := options.OnToken(inference.TokenEvent{Piece: piece}); err != nil {
				return nil, "", err
			}
		}
		ids = append(ids, tokenizer.TokenID(len(words)+index))
	}
	return ids, "", nil
}

// TestRecordExactSuiteReplays pins the bootstrap golden: what the model
// produced becomes the expected text, with the prompt and generated token
// counts a compiled exact plan requires, so a replay of the same model
// reproduces it exactly.
func TestRecordExactSuiteReplays(t *testing.T) {
	suite := testcheck.Must(RecordExactSuite(t.Context(), recordingGenerator{}, "intake-test", []string{"one two three", "alpha beta"}, 4))(t)
	if len(suite.Cases) != 2 || suite.Cases[0].Text != "three two one " || suite.Cases[0].PromptTokens != 3 || suite.Cases[0].GeneratedTokens != 3 {
		t.Fatalf("recorded suite = %+v", suite.Cases)
	}
	if _, err := evaluation.CompileExact(suite); err != nil {
		t.Fatalf("recorded suite does not compile: %v", err)
	}
	if _, err := RecordExactSuite(t.Context(), recordingGenerator{}, "", nil, 0); err == nil {
		t.Fatal("an empty recording request was accepted")
	}
}

func TestMeasuredVerificationRequestBinding(t *testing.T) {
	ctx := t.Context()
	path := t.TempDir()
	store := overgodbtest.Create(t, path)
	model := testutil.ArtifactID(t, artifact.KindModel, "verification request fixture")
	testutil.PublishArtifact(t, store, model)
	definition := testcheck.Must(modelrecipe.CapabilityDefinition(recipe.TaskSeq2Seq, model))(t)
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/candidate", definition); err != nil {
		t.Fatal(err)
	}
	const revision = "0123456789abcdef0123456789abcdef01234567"
	var inputs []artifact.Content
	var records []modelrecipe.Verification
	for _, text := range []string{"first request", "second request"} {
		input := testcheck.Must(artifact.JSONContent(artifact.JSONContract(artifact.KindFile, "overgo.verification-request-fixture.v1"), text))(t)
		measured := capabilityruntime.Measured{Input: input}
		verified := testcheck.Must(PublishMeasuredVerification(ctx, store, definition, revision, time.Millisecond, "host", "go", "fixture execution", measured))(t)
		bound, err := runrecord.VerifyGateRun(ctx, store, definition.ID, verified.Gate, verified.Run)
		if err != nil || !slices.Equal(bound.Run.Inputs, []artifact.ID{input.Descriptor.ID}) {
			t.Fatalf("request lost from verifier run: %+v: %v", bound.Run.Inputs, err)
		}
		head, sequence := store.Head()
		replayed, err := PublishMeasuredVerification(ctx, store, definition, revision, time.Millisecond, "host", "go", "fixture execution", measured)
		after, afterSequence := store.Head()
		if err != nil || replayed != verified || head != after || sequence != afterSequence {
			t.Fatalf("replay repeated publication: %+v versus %+v: %v", replayed, verified, err)
		}
		inputs, records = append(inputs, input), append(records, verified)
	}
	if records[0].Run == records[1].Run {
		t.Fatal("different requests with identical measurements collided")
	}
	tampered := inputs[0].Clone()
	tampered.Data[0] ^= 1
	for _, bad := range []artifact.Content{
		tampered, {Descriptor: inputs[0].Descriptor}, {Data: inputs[0].Data},
		{Descriptor: artifact.Descriptor{Schema: "missing identity"}}, {Data: []byte{}},
	} {
		head, sequence := store.Head()
		_, err := PublishMeasuredVerification(ctx, store, definition, revision, time.Millisecond, "host", "go", "fixture execution", capabilityruntime.Measured{Input: bad})
		after, afterSequence := store.Head()
		if err == nil || head != after || sequence != afterSequence {
			t.Fatalf("invalid request admitted or partially published: %v", err)
		}
	}
	legacy := testcheck.Must(PublishVerification(ctx, store, definition, revision, time.Millisecond, "host", "go", "fixture execution"))(t)
	legacyRun, err := runrecord.RequireExactRun(ctx, store, legacy.Run)
	if err != nil || len(legacyRun.Inputs) != 0 {
		t.Fatalf("legacy publication invented a request: %v", err)
	}
	testcheck.NoError(t, store.Close())
	reopened := overgodbtest.OpenReadOnly(t, path)
	for i, verified := range records {
		run, err := runrecord.RequireExactRun(ctx, reopened, verified.Run)
		if err != nil || !slices.Equal(run.Inputs, []artifact.ID{inputs[i].Descriptor.ID}) {
			t.Fatalf("request lineage lost after reopen: %v", err)
		}
		content, found, err := artifact.ReadContent(ctx, reopened, inputs[i].Descriptor.ID)
		if err != nil || !found || content.Descriptor != inputs[i].Descriptor || !bytes.Equal(content.Data, inputs[i].Data) {
			t.Fatalf("request content lost after reopen: %v", err)
		}
	}
}
