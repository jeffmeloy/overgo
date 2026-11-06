//go:build modeltest

package seq2seq

import (
	"encoding/json"
	"os"
	"slices"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/textgeneration"
)

// upstreamCallsScript writes the fixture this test reads, from the model's
// own code. It is kept where Python is kept, on the machine that ran it and
// not in git: the fixture is the data.
const upstreamCallsScript = "scripts/python_golden_references/needle_upstream_calls.py"

// upstreamCalls mirrors fixtures/needle_upstream_calls.json: for each case of
// the function-call evaluation, what needle's upstream code builds and answers.
type upstreamCalls struct {
	GeneratedTokens int `json:"generated_tokens"`
	Cases           []struct {
		Name            string `json:"name"`
		Query           string `json:"query"`
		Tools           string `json:"tools"`
		EncoderIDs      []int  `json:"encoder_ids"`
		NamedEncoderIDs []int  `json:"named_encoder_ids"`
		PlainFloat32    string `json:"plain_f32"`
		NamedFloat32    string `json:"named_f32"`
		Named           string `json:"named"`
		Served          string `json:"served"`
	} `json:"cases"`
}

// TestNeedleCallsMatchUpstreamOnEvaluationCases holds the path a served
// request takes, from its text to the call it answers with, to the model's
// own code on the cases the model is scored on. The encoder's input is the
// ids that code builds, for the tools as given and for the tools as a served
// request renames them: the tokenizer model declares byte-pair merges, and
// segmented as a unigram table no request's ids were the model's. The answer
// is the answer that code returns, to the character, for the plain decode and
// for a served request, when it computes as overgo does: on the bfloat16
// parameters it loads, which the shipped weights file holds, with the
// arithmetic in float32. That code's own arithmetic is bfloat16 throughout,
// and no port that computes otherwise can match it to the character: the test
// says how many of its answers turn on that alone. What is not held is that
// code's constrained decoding, which a served request here does not do; the
// test says how many answers that changes too.
func TestNeedleCallsMatchUpstreamOnEvaluationCases(t *testing.T) {
	raw, err := os.ReadFile(testutil.FixturePath(t, "needle_upstream_calls.json"))
	if err != nil {
		t.Fatalf("the upstream calls are absent (%v); write them with %s", err, upstreamCallsScript)
	}
	var fixture upstreamCalls
	testcheck.NoError(t, json.Unmarshal(raw, &fixture))
	if len(fixture.Cases) == 0 || fixture.GeneratedTokens <= 0 {
		t.Fatalf("the fixture holds %d cases of %d tokens", len(fixture.Cases), fixture.GeneratedTokens)
	}
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	generator, err := LoadGenerator(roots.ResolveModelPath("needle"))
	if err != nil {
		t.Fatalf("Needle artifact unavailable: %v", err)
	}
	if generator.merges == nil {
		t.Fatal("the tokenizer model's byte-pair merges were not read, and its text would be segmented as a unigram table")
	}
	arithmetic, constrained := 0, 0
	for _, reference := range fixture.Cases {
		prompt := reference.Query + " " + toolsMarker + " " + reference.Tools
		given, _, err := generator.source(prompt, false)
		testcheck.NoError(t, err)
		if !slices.Equal(given, reference.EncoderIDs) {
			t.Fatalf("%s: the encoder input for the tools as given is %v, and upstream builds %v", reference.Name, given, reference.EncoderIDs)
		}
		renamed, _, err := generator.source(prompt, true)
		testcheck.NoError(t, err)
		if !slices.Equal(renamed, reference.NamedEncoderIDs) {
			t.Fatalf("%s: the encoder input a served request builds is %v, and upstream builds %v", reference.Name, renamed, reference.NamedEncoderIDs)
		}
		// The plain decode: the ids of the tools as given, and no names to restore.
		encoded := testcheck.Must(generator.model.encodeTokens(given, fixture.GeneratedTokens))(t)
		selector := testcheck.Must(generator.prepareGeneration(encoded))(t)
		plain := testcheck.Must(selector.selectText())(t)
		if plain != reference.PlainFloat32 {
			t.Fatalf("%s: the plain decode answers\n  %q\nand upstream, computing in float32,\n  %q", reference.Name, plain, reference.PlainFloat32)
		}
		served := testcheck.Must(generator.Generate(textgeneration.Request{Text: prompt, MaxTokens: fixture.GeneratedTokens}))(t)
		if served != reference.NamedFloat32 {
			t.Fatalf("%s: a served request answers\n  %q\nand upstream, computing in float32 with names normalized,\n  %q", reference.Name, served, reference.NamedFloat32)
		}
		if reference.Named != reference.NamedFloat32 {
			arithmetic++
		}
		if reference.Served != reference.Named {
			constrained++
		}
	}
	t.Logf("%d cases: overgo's encoder input and its answers are upstream's, computing in float32, in every one; %d of upstream's own answers turn on its bfloat16 arithmetic, and its constrained decoding, which is not done here, changes %d", len(fixture.Cases), arithmetic, constrained)
}
