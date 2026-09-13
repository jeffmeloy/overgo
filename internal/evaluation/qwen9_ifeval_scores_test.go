package evaluation

import (
	"maps"
	"slices"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func TestQwenNineIFEvalScoresAcceptance(t *testing.T) {
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	parse := func(value string) artifact.ID {
		t.Helper()
		id, err := artifact.ParseID(value)
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	// The immutable request fixes the producer, model, recipe and cumulative budget.
	requestID := parse("evidence:sha256:4e5eb501233489e0e4c54289971d5a67f729a5cf1c1153ed5d31e3eb2bd9c00e")
	var request struct {
		Producer            string
		ModelDefinition     artifact.ID `json:"model_definition"`
		Recipe              artifact.ID
		Cases, Instructions int
		Started             time.Time `json:"started_utc"`
		Deadline            time.Time `json:"deadline_utc"`
		BudgetSeconds       int       `json:"budget_seconds"`
	}
	readRetainedEvidence(t, store, requestID, &request)
	completionID, found, err := store.ResolveAlias(t.Context(), "evaluation/qwen9-ifeval-cc267157/completion")
	if err != nil || !found {
		t.Fatalf("Qwen9 completion absent: found=%t err=%v", found, err)
	}
	var completion struct {
		Request     artifact.ID
		Started     time.Time `json:"process_started_utc"`
		CompletedBy time.Time `json:"completed_by_utc"`
		ExitCode    *int      `json:"exit_code"`
	}
	readRetainedEvidence(t, store, completionID, &completion)
	if completion.Request != requestID || completion.ExitCode == nil || *completion.ExitCode != 0 ||
		request.BudgetSeconds <= 0 || request.Deadline.Sub(request.Started) != time.Duration(request.BudgetSeconds)*time.Second ||
		completion.Started.Before(request.Started) || completion.CompletedBy.Before(completion.Started) || completion.CompletedBy.After(request.Deadline) {
		t.Fatal("acquisition did not complete within its original cumulative budget")
	}
	nativeID, found, err := store.ResolveAlias(t.Context(), "evaluation/qwen9-ifeval-cc267157/native")
	if err != nil || !found {
		t.Fatalf("Qwen9 native scores absent: found=%t err=%v", found, err)
	}
	checkIFEvalNativeReference(t, store, nativeID)
	var scores struct {
		ifevalNativeScores
		Evidence artifact.ID `json:"evaluation_evidence"`
	}
	readRetainedEvidence(t, store, nativeID, &scores)
	record, err := RequireEvaluationEvidence(t.Context(), store, scores.Evidence)
	if err != nil {
		t.Fatal(err)
	}
	if record.CodeCommit != request.Producer || record.ModelDefinition != request.ModelDefinition ||
		record.Recipe != request.Recipe || record.Report != scores.Selection ||
		scores.Prompts != request.Cases || scores.Instructions != request.Instructions {
		t.Fatal("quality evidence differs from the frozen acquisition request")
	}
	plan, err := loadEvidencePlan(t.Context(), store, record.Plan)
	if err != nil {
		t.Fatal(err)
	}
	protocol, err := ReadExecutionPolicy(t.Context(), store, plan.Execution())
	if err != nil || protocol.Prompting != PromptingChatTemplate || plan.body.CaseProfile != scores.Profile {
		t.Fatalf("instruction protocol or case profile differs: %v", err)
	}
	var report InstructionRulesReport
	readRetainedEvidence(t, store, record.Report, &report)
	var suite InstructionRulesSuite
	readRetainedEvidence(t, store, plan.body.CaseProfile, &suite)
	if len(report.Observations) != request.Cases || len(suite.Cases) != request.Cases || report.Plan != record.Plan {
		t.Fatal("complete report or case denominator differs")
	}
	observations := map[string]InstructionRulesObservation{}
	responses := map[string]string{}
	for _, row := range report.Observations {
		if _, duplicate := observations[row.Name]; duplicate {
			t.Fatal("duplicate instruction response")
		}
		observations[row.Name] = row
		responses[row.Name] = row.Raw
	}
	// Native IFEval uses the retained 1280-token protocol, including capped outputs.
	for _, testCase := range suite.Cases {
		row, found := observations[testCase.Name]
		if !found || testCase.MaxTokens != 1280 || row.Generated < 0 || row.Generated > testCase.MaxTokens || row.Prompt <= 0 {
			t.Fatalf("response or native token bound differs: %s", testCase.Name)
		}
		caseID, err := artifact.JSONID(artifact.KindDatasetShard, testCase)
		if err != nil {
			t.Fatal(err)
		}
		alias := "evaluation/instruction-generations/" + plan.Identity().String() + "/" + caseID.String()
		saved, found, err := loadInstructionGeneration(t.Context(), store, alias, plan, caseID, testCase)
		if err != nil || !found || saved.Result.Text != row.Raw || saved.Result.PromptTokens != row.Prompt || saved.Result.GeneratedTokens != row.Generated {
			t.Fatalf("report differs from durable generation: %s found=%t err=%v", testCase.Name, found, err)
		}
	}
	for _, native := range scores.Rows {
		row, found := observations[native.Name]
		if !found || !slices.Equal(row.Strict, native.Strict) || !slices.Equal(row.Loose, native.Loose) {
			t.Errorf("published verdicts differ from native: %s strict=%v/%v loose=%v/%v", native.Name, row.Strict, native.Strict, row.Loose, native.Loose)
		}
	}
	metrics := map[string]float64{
		"prompt_strict": report.PromptStrict, "prompt_loose": report.PromptLoose,
		"instruction_strict": report.InstructionStrict, "instruction_loose": report.InstructionLoose,
	}
	if !maps.Equal(metrics, scores.Metrics) {
		t.Fatalf("published metrics differ from native: got=%v want=%v", metrics, scores.Metrics)
	}
	checkRetainedIFEvalScores(t, store, scores.ifevalNativeScores, responses)
}
