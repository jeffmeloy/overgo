package evaluation

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/tokenizer"
)

type interruptedInstructionFixture struct {
	instructionChatFixture
	stopPrompt string
}

func (f *interruptedInstructionFixture) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	if prompt == f.stopPrompt {
		return nil, "", context.Canceled
	}
	return f.instructionChatFixture.Generate(ctx, prompt, options)
}

func TestInstructionResumeCounterexample(t *testing.T) {
	t.Parallel()
	suite := instructionRulesSuite{Kind: instructionRulesKind, Schema: "fixture/v1", Source: "resume-fixture", Cases: []instructionRulesCase{
		{Name: "first", Prompt: "First answer.", MaxTokens: 1, Rules: []instructionRule{{Name: "uppercase", Kind: ruleUppercase}}},
		{Name: "second", Prompt: "Second answer.", MaxTokens: 1, Rules: []instructionRule{{Name: "uppercase", Kind: ruleUppercase}}},
	}}
	compiled, err := compileInstructionRules(suite)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := bindInstructionRules(compiled, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"), RuntimeRecipe: planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "environment"),
		Execution: ExecutionPolicy{Lifecycle: LifecycleResident, Prompting: PromptingChatTemplate},
	})
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	publishPlanFixtureAuthorities(t, store, plan)
	interrupted := &interruptedInstructionFixture{instructionChatFixture: instructionChatFixture{answer: "OK"}, stopPrompt: "<user>Second answer.<model>"}
	_, runErr := evaluateInstructionRules(t.Context(), store, interrupted, compiled, plan)
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if !errors.Is(runErr, context.Canceled) || len(interrupted.prompts) != 1 {
		t.Fatalf("first attempt: err=%v prompts=%v", runErr, interrupted.prompts)
	}
	store, err = overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	resumed := &instructionChatFixture{answer: "OK"}
	report, err := evaluateInstructionRules(t.Context(), store, resumed, compiled, plan)
	if err != nil {
		t.Fatal(err)
	}
	if len(resumed.prompts) != 1 || resumed.prompts[0] != "<user>Second answer.<model>" || len(report.Observations) != len(suite.Cases) {
		t.Fatalf("restart reacquired completed work: generated=%v report_cases=%d", resumed.prompts, len(report.Observations))
	}
}

// instructionCommitFixture stands between an evaluation and its store at the
// commits that write answers down: before the first of them it runs
// beforeCommit, once, and it fails each of them with commitErr when that is
// set. Every other commit goes through.
type instructionCommitFixture struct {
	artifact.Repository
	beforeCommit func()
	commitErr    error
}

func (f *instructionCommitFixture) Commit(ctx context.Context, batch artifact.Batch) (artifact.CommitID, error) {
	if !strings.HasPrefix(batch.Key, observedKey) {
		return f.Repository.Commit(ctx, batch)
	}
	if f.beforeCommit != nil {
		before := f.beforeCommit
		f.beforeCommit = nil
		before()
	}
	if f.commitErr != nil {
		return artifact.CommitID{}, f.commitErr
	}
	return f.Repository.Commit(ctx, batch)
}

type cancellingInstructionFixture struct {
	instructionChatFixture
	cancel context.CancelCauseFunc
}

func (f *cancellingInstructionFixture) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	ids, text, err := f.instructionChatFixture.Generate(ctx, prompt, options)
	f.cancel(context.Canceled)
	return ids, text, err
}

func instructionResumeFixture(t *testing.T) (*overgodb.Store, instructionRulesPlan, Plan) {
	t.Helper()
	// One generated token is sufficient to witness uppercase success or failure.
	compiled, err := compileInstructionRules(instructionRulesSuite{Kind: instructionRulesKind, Schema: "fixture/v1", Source: "resume-fixture", Cases: []instructionRulesCase{
		{Name: "first", Prompt: "First answer.", MaxTokens: 1, Rules: []instructionRule{{Name: "uppercase", Kind: ruleUppercase}}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := bindInstructionRules(compiled, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"), RuntimeRecipe: planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "environment"),
		Execution: ExecutionPolicy{Lifecycle: LifecycleResident, Prompting: PromptingChatTemplate},
	})
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	})
	publishPlanFixtureAuthorities(t, store, plan)
	contents, err := datasetContents(compiled.dataset, compiled.split, compiled.suite.Cases, instructionRulesDatasetContract, instructionRulesSplitContract)
	if err != nil {
		t.Fatal(err)
	}
	if err := publishPlanAuthorities(t.Context(), store, plan, contents); err != nil {
		t.Fatal(err)
	}
	return store, compiled, plan
}

func TestInstructionResumeRetainsQualityFailure(t *testing.T) {
	t.Parallel()
	store, compiled, plan := instructionResumeFixture(t)
	failed := &instructionChatFixture{answer: "lowercase"}
	first, err := evaluateInstructionRules(t.Context(), store, failed, compiled, plan)
	if err != nil {
		t.Fatal(err)
	}
	improved := &instructionChatFixture{answer: "UPPERCASE"}
	replay, err := evaluateInstructionRules(t.Context(), store, improved, compiled, plan)
	if err != nil {
		t.Fatal(err)
	}
	if first.PromptStrict != 0 || replay.ID != first.ID || len(improved.prompts) != 0 {
		t.Fatalf("replaced retained failure: first=%+v replay=%+v generated=%v", first, replay, improved.prompts)
	}
}

func TestInstructionResumeFinishesPublicationAfterCancellation(t *testing.T) {
	t.Parallel()
	store, compiled, plan := instructionResumeFixture(t)
	ctx, cancel := context.WithCancelCause(t.Context())
	defer cancel(context.Canceled)
	generator := &cancellingInstructionFixture{instructionChatFixture: instructionChatFixture{answer: "OK"}, cancel: cancel}
	_, err := evaluateInstructionRules(ctx, store, generator, compiled, plan)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation = %v", err)
	}
	resumed := &instructionChatFixture{answer: "WRONG"}
	report, err := evaluateInstructionRules(t.Context(), store, resumed, compiled, plan)
	if err != nil {
		t.Fatal(err)
	}
	if len(resumed.prompts) != 0 || report.Observations[0].Raw != "OK" {
		t.Fatalf("completed response lost: %+v prompts=%v", report, resumed.prompts)
	}
}

func TestInstructionResumePublicationFailure(t *testing.T) {
	t.Parallel()
	store, compiled, plan := instructionResumeFixture(t)
	failure := errors.New("fixture storage failure")
	repository := &instructionCommitFixture{Repository: store, commitErr: failure}
	_, err := evaluateInstructionRules(t.Context(), repository, &instructionChatFixture{answer: "OK"}, compiled, plan)
	if !errors.Is(err, failure) {
		t.Fatalf("publication error = %v", err)
	}
	resumed := &instructionChatFixture{answer: "OK"}
	if _, err := evaluateInstructionRules(t.Context(), store, resumed, compiled, plan); err != nil {
		t.Fatal(err)
	}
	if len(resumed.prompts) != 1 {
		t.Fatal("unpublished result received reuse credit")
	}
}

func TestInstructionResumeConcurrentPublication(t *testing.T) {
	t.Parallel()
	for _, same := range []bool{true, false} {
		t.Run(map[bool]string{true: "same", false: "different"}[same], func(t *testing.T) {
			store, compiled, plan := instructionResumeFixture(t)
			winner := &instructionChatFixture{answer: "OK"}
			answer := winner.answer
			if !same {
				answer = "DIFFERENT"
			}
			// Another evaluation of the plan writes its answer down after this one's model has answered and before this one writes.
			repository := &instructionCommitFixture{Repository: store, beforeCommit: func() {
				if _, err := evaluateInstructionRules(t.Context(), store, winner, compiled, plan); err != nil {
					t.Fatal(err)
				}
			}}
			_, err := evaluateInstructionRules(t.Context(), repository, &instructionChatFixture{answer: answer}, compiled, plan)
			if same && err != nil || !same && err == nil {
				t.Fatalf("same=%v error=%v", same, err)
			}
			replay := &instructionChatFixture{answer: "REPLACEMENT"}
			report, err := evaluateInstructionRules(t.Context(), store, replay, compiled, plan)
			if err != nil {
				t.Fatal(err)
			}
			if report.Observations[0].Raw != winner.answer || len(replay.prompts) != 0 {
				t.Fatalf("winner replaced: %+v", report.Observations[0])
			}
		})
	}
}

func TestInstructionResumeRejectsDifferentBindings(t *testing.T) {
	t.Parallel()
	for _, field := range []string{"model", "recipe", "environment", "source", "protocol", "prompt", "token-cap"} {
		t.Run(field, func(t *testing.T) {
			store, compiled, plan := instructionResumeFixture(t)
			if _, err := evaluateInstructionRules(t.Context(), store, &instructionChatFixture{answer: "OK"}, compiled, plan); err != nil {
				t.Fatal(err)
			}
			authorities := ExactAuthorities{
				ModelDefinition: plan.body.ModelDefinition, RuntimeRecipe: plan.body.RuntimeRecipe,
				CodeCommit: plan.body.CodeCommit, Environment: plan.body.Environment,
				Execution: ExecutionPolicy{Lifecycle: LifecycleResident, Prompting: PromptingChatTemplate},
			}
			switch field {
			case "model":
				authorities.ModelDefinition = planID(t, artifact.KindModelDefinition, "changed-model")
			case "recipe":
				authorities.RuntimeRecipe = planID(t, artifact.KindRecipe, "changed-recipe")
			case "environment":
				authorities.Environment = planID(t, artifact.KindEvidence, "changed-environment")
			case "source":
				authorities.CodeCommit = "1111111111111111111111111111111111111111"
			case "protocol":
				authorities.Execution.Prompting = PromptingRawCompletion
			case "prompt":
				compiled.suite.Cases[0].Prompt = "Different question."
			case "token-cap":
				compiled.suite.Cases[0].MaxTokens++
			}
			changed, err := compileInstructionRules(compiled.suite)
			if err != nil {
				t.Fatal(err)
			}
			changedPlan, err := bindInstructionRules(changed, authorities)
			if err != nil {
				t.Fatal(err)
			}
			if changedPlan.identity == plan.identity {
				t.Fatal("changed authority did not alter plan")
			}
			if field == "model" || field == "recipe" || field == "environment" {
				publishPlanFixtureAuthorities(t, store, changedPlan)
			}
			generator := &instructionChatFixture{answer: "NEW"}
			if _, err := evaluateInstructionRules(t.Context(), store, generator, changed, changedPlan); err != nil {
				t.Fatal(err)
			}
			if len(generator.prompts) != 1 {
				t.Fatalf("reused different %s", field)
			}
		})
	}
}

func TestInstructionResumeRejectsMisboundRecord(t *testing.T) {
	t.Parallel()
	store, compiled, plan := instructionResumeFixture(t)
	testCase := compiled.suite.Cases[0]
	// What the plan's execution is asked, named as the evaluation names it.
	observed, err := plan.observed([]struct {
		Name      string `json:"name"`
		Prompt    string `json:"prompt"`
		MaxTokens int    `json:"max_tokens"`
	}{{testCase.Name, testCase.Prompt, testCase.MaxTokens}})
	if err != nil {
		t.Fatal(err)
	}
	// A record that says it answers something else is filed under it.
	answer, err := json.Marshal(ExactResult{Name: testCase.Name, PromptTokens: 1, GeneratedTokens: 1, Text: "OK"})
	if err != nil {
		t.Fatal(err)
	}
	misbound, err := caseAnswersCodec.NewInitial(caseAnswers{Observed: planID(t, artifact.KindProfile, "asked-of-another"), Cases: []json.RawMessage{answer}})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := caseAnswersCodec.Batch("misbound-answers", misbound, artifact.DependencyLineage(misbound.ID, observed), nil)
	if err != nil {
		t.Fatal(err)
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: observed})
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	generator := &instructionChatFixture{answer: "OK"}
	if _, err := evaluateInstructionRules(t.Context(), store, generator, compiled, plan); err == nil {
		t.Fatal("accepted misbound record")
	}
	if len(generator.prompts) != 0 {
		t.Fatal("treated invalid state as cache miss")
	}
}
