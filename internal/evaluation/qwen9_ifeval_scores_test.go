package evaluation

import (
	"maps"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func TestQwenNineIFEvalScoresAcceptance(t *testing.T) {
	checkFullNativeIFEvalScores(t,
		"evidence:sha256:4e5eb501233489e0e4c54289971d5a67f729a5cf1c1153ed5d31e3eb2bd9c00e",
		"evaluation/qwen9-ifeval-cc267157", "", "")
}

func TestNativeIFEvalInputControl(t *testing.T) {
	checkFullNativeIFEvalScores(t,
		"evidence:sha256:4e5eb501233489e0e4c54289971d5a67f729a5cf1c1153ed5d31e3eb2bd9c00e",
		"evaluation/qwen9-ifeval-cc267157",
		"evidence:sha256:bf56eb0c6495d306b2b4c7b59c59db259d1ef32f298ccfe5e672db5ab5f7f3d1", "")
}

func TestGemmaNativeIFEvalAcceptance(t *testing.T) {
	t.Run("reference-control", TestNativeIFEvalInputControl)
	var fixtures map[string]struct {
		Request artifact.ID `json:"request"`
		Native  artifact.ID `json:"native"`
		Prefix  string      `json:"prefix"`
	}
	root := testutil.RepoRoot(t)
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs/verification/gemma12-ifeval-acceptance.json"), &fixtures); err != nil {
		t.Fatal(err)
	}
	// These are the two registered execution artifacts in this model bundle.
	models := map[string]string{
		"native": "model:sha256:d0be4bb0e24f72ee0589f422ef579115068a7602837bdb4a674f5baaacf030ba",
		"bf16":   "model:sha256:f5e632cd3f6050ab8ae06df55766de5737282689d211f3e2f172db52ff671401",
	}
	t.Run("coverage", func(t *testing.T) {
		if len(fixtures) != len(models) {
			t.Fatal("Gemma execution-variant denominator differs")
		}
	})
	for variant, model := range models {
		t.Run(variant, func(t *testing.T) {
			fixture, found := fixtures[variant]
			if !found || fixture.Request.Kind() != artifact.KindEvidence || fixture.Native.Kind() != artifact.KindEvidence || fixture.Prefix == "" {
				t.Fatal("Gemma native acceptance is missing a bound request or result")
			}
			checkFullNativeIFEvalScores(t, fixture.Request.String(), fixture.Prefix, fixture.Native.String(), model)
		})
	}
}

func checkFullNativeIFEvalScores(t *testing.T, requestText, prefix, nativeText, expectedModel string) {
	t.Helper()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
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
	requestID := parse(requestText)
	var request struct {
		Producer            string
		Model               artifact.ID
		ModelDefinition     artifact.ID `json:"model_definition"`
		Recipe              artifact.ID
		Cases, Instructions int
		Started             time.Time   `json:"started_utc"`
		Deadline            time.Time   `json:"deadline_utc"`
		BudgetSeconds       int         `json:"budget_seconds"`
		CampaignAnchor      artifact.ID `json:"campaign_anchor"`
		InitialCompletion   artifact.ID `json:"initial_completion"`
		ResumeInventory     artifact.ID `json:"resume_inventory"`
	}
	readRetainedEvidence(t, store, requestID, &request)
	if expectedModel != "" && request.Model.String() != expectedModel {
		t.Fatal("IFEval request names another execution artifact")
	}
	completionID, found, err := store.ResolveAlias(t.Context(), prefix+"/completion")
	if err != nil || !found {
		t.Fatalf("IFEval completion absent: found=%t err=%v", found, err)
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
	if request.CampaignAnchor.Valid() {
		var anchor struct {
			Producer string
			Started  time.Time `json:"started_utc"`
		}
		var terminal struct {
			Request         artifact.ID
			BudgetExhausted bool `json:"budget_exhausted"`
		}
		readRetainedEvidence(t, store, request.CampaignAnchor, &anchor)
		readRetainedEvidence(t, store, request.InitialCompletion, &terminal)
		// cmd/evaluate's existing full-pass ceiling includes failed attempts and waits.
		if !request.Started.Equal(anchor.Started) || request.Producer != anchor.Producer ||
			request.Deadline.Sub(anchor.Started) > 6*time.Hour ||
			terminal.Request != request.CampaignAnchor || !terminal.BudgetExhausted {
			t.Fatal("continuation reset its source, cumulative clock or terminal predecessor")
		}
	}
	var nativeID artifact.ID
	if nativeText != "" {
		nativeID = parse(nativeText)
	} else {
		nativeID, found, err = store.ResolveAlias(t.Context(), prefix+"/native")
		if err != nil || !found {
			t.Fatalf("IFEval native scores absent: found=%t err=%v", found, err)
		}
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
	if request.ResumeInventory.Valid() {
		var retained struct {
			Request, Completion, Plan artifact.ID
			Outputs                   map[string]artifact.ID
		}
		readRetainedEvidence(t, store, request.ResumeInventory, &retained)
		if retained.Request != request.CampaignAnchor || retained.Completion != request.InitialCompletion ||
			retained.Plan != record.Plan || len(retained.Outputs) == 0 || len(retained.Outputs) > request.Cases {
			t.Fatal("resume inventory differs from the original acquisition")
		}
		for caseID, outputID := range retained.Outputs {
			alias := "evaluation/instruction-generations/" + record.Plan.String() + "/" + caseID
			current, found, err := store.ResolveAlias(t.Context(), alias)
			if err != nil || !found || current != outputID {
				t.Fatalf("continuation replaced a completed response: %s found=%t err=%v", caseID, found, err)
			}
		}
		t.Logf("continuation preserved %d exact completed responses", len(retained.Outputs))
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
