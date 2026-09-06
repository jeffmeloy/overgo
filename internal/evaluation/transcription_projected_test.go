package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

func TestProjectedTranscriptionBindsPromptAndBudget(t *testing.T) {
	fixture := singleTranscriptionEvaluationFixture(t, TranscriptionCase{
		Name: "pinned", Group: "validation", Source: audioReference(t, "projected-protocol"),
		Reference: "reference stays outside prediction", SampleCount: 16000, SampleRate: 16000,
	})
	defer fixture.store.Close()
	suite := fixture.compiled.suite
	for _, invalid := range []struct {
		prompt  string
		maximum int
	}{{"", 8}, {" ", 8}, {"transcribe", 0}, {"transcribe", -1}} {
		suite.Prompt, suite.MaxTokens = invalid.prompt, invalid.maximum
		if _, err := CompileTranscription(suite); err == nil {
			t.Fatalf("unbound protocol accepted: %+v", invalid)
		}
	}
	suite.Prompt, suite.MaxTokens = "Transcribe the speech exactly.", 128
	suite.DecodeRecipe = testutil.ArtifactID(t, artifact.KindRecipe, "exact decoder")
	first, err := CompileTranscription(suite)
	if err != nil {
		t.Fatal(err)
	}
	if first.identity == fixture.compiled.identity {
		t.Fatal("projection protocol reused the native speech suite identity")
	}
	suite.MaxTokens++
	second, err := CompileTranscription(suite)
	if err != nil {
		t.Fatal(err)
	}
	if first.identity == second.identity {
		t.Fatal("token budget is absent from the suite identity")
	}
	suite.Prompt += " Return only the transcript."
	third, err := CompileTranscription(suite)
	if err != nil {
		t.Fatal(err)
	}
	if second.identity == third.identity {
		t.Fatal("prompt is absent from the suite identity")
	}
	suite.DecodeRecipe = testutil.ArtifactID(t, artifact.KindRecipe, "different decoder")
	fourth, err := CompileTranscription(suite)
	if err != nil || fourth.identity == third.identity {
		t.Fatalf("decoder identity is not bound: %v", err)
	}
	authorities := ExactAuthorities{ModelDefinition: fixture.plan.body.ModelDefinition, RuntimeRecipe: fixture.recipe,
		CodeCommit: transcriptionTestCommit, Environment: fixture.environment, Execution: ExecutionPolicy{Lifecycle: LifecycleResident}}
	if _, err := BindTranscription(fourth, authorities); err == nil {
		t.Fatal("projected suite accepted raw-completion prompting")
	}
	authorities.Execution.Prompting = PromptingChatTemplate
	plan, err := BindTranscription(fourth, authorities)
	if err != nil {
		t.Fatal(err)
	}
	prediction := fixture.successfulPrediction(t, "pinned", "reference stays outside prediction")
	if _, err := EvaluateTranscription(t.Context(), fixture.store, fourth, plan, []TranscriptionPrediction{prediction}); err == nil {
		t.Fatal("unbound native speech run was admitted as projected transcription")
	}
	suite.DecodeRecipe = artifact.ID{}
	if _, err := CompileTranscription(suite); err == nil {
		t.Fatal("projected transcription accepted an unbound decoder")
	}
}

func TestProjectedTranscriptionRefusesTruncation(t *testing.T) {
	stopIDs := []tokenizer.TokenID{9, 10}
	for _, test := range []struct {
		tokens   []tokenizer.TokenID
		maximum  int
		complete bool
	}{
		{nil, 2, false},
		{[]tokenizer.TokenID{1, 2}, 2, false},
		{[]tokenizer.TokenID{9, 1}, 2, false},
		{[]tokenizer.TokenID{1, 9}, 2, true},
		{[]tokenizer.TokenID{1, 10}, 2, true},
		{[]tokenizer.TokenID{1, 2, 9}, 2, false},
	} {
		if got := completeProjectedTranscript(test.tokens, stopIDs, test.maximum) == nil; got != test.complete {
			t.Fatalf("tokens=%v maximum=%d complete=%t, want %t", test.tokens, test.maximum, got, test.complete)
		}
	}
}
