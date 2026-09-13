package main

import (
	"encoding/json"
	"maps"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// The compact result references the complete acquisition and its passed scorer
// gate. This check reads historical evidence; it never executes a model.
func TestMiniCPMIFEvalGoParityAcceptance(t *testing.T) {
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	parse := func(text string) artifact.ID {
		t.Helper()
		id, err := artifact.ParseID(text)
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	read := func(id artifact.ID, target any) {
		t.Helper()
		content, err := artifact.RequireTypedContent(t.Context(), store, id)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(content.Data, target); err != nil {
			t.Fatal(err)
		}
	}
	var parity struct {
		ScorerCommit          string            `json:"scorer_commit"`
		GatePreparation       artifact.ID       `json:"gate_preparation"`
		GateFinalization      artifact.ID       `json:"gate_finalization"`
		GateResult            artifact.ID       `json:"gate_result"`
		NativeScores          artifact.ID       `json:"native_scores"`
		ScoreRows             artifact.ID       `json:"score_rows"`
		RawSelection          artifact.ID       `json:"raw_selection"`
		AcquisitionProfile    artifact.ID       `json:"acquisition_profile"`
		NativeParameters      artifact.ID       `json:"native_parameters"`
		CompiledProfiles      map[string]string `json:"compiled_profiles_sha256"`
		Verify                string
		Prompts, Instructions int
		PairedVerdicts        int `json:"paired_verdicts"`
		Mismatches            int
		NewInferenceCalls     int `json:"new_inference_calls"`
		ModelLoads            int `json:"model_loads"`
		Correct               map[string]int
		Metrics               map[string]float64
	}
	read(parse("evidence:sha256:3b5154d8426709ae51e918ea35a0b603e86606d7efd11a99d1ba246581a54555"), &parity)
	if parity.Prompts != 541 || parity.Instructions != 834 || parity.PairedVerdicts != 1668 || parity.Mismatches != 0 || parity.NewInferenceCalls != 0 || parity.ModelLoads != 0 || parity.AcquisitionProfile.String() != miniIFEvalProfile || parity.NativeScores.String() != "evidence:sha256:78a1241f82ce179d229248564ccffb66f291d922eba864d1a261fc94d386c839" || parity.NativeParameters.String() != "profile:sha256:78fff5aa9eb2bc5015aadba767afe49003cb813117422ecfbfa4fb65a19b8a4f" {
		t.Fatal("IFEval retained parity denominator or acquisition differs")
	}
	if !maps.Equal(parity.Correct, map[string]int{"prompt_strict": 365, "prompt_loose": 381, "instruction_strict": 642, "instruction_loose": 661}) || !maps.Equal(parity.CompiledProfiles, map[string]string{
		"internal/evaluation/testdata/ifeval_langdetect_profile.json": "2b9af36fa26af89c5a87e5eae7b0b989236c06a8bb9f71f843d73e1676fc1c7b",
		"internal/evaluation/testdata/ifeval_nltk_profile.json":       "3d2055c31e46ee2542569ceab92d2e7beabe336b196e4b7698662e4fe1c23892",
	}) {
		t.Fatal("IFEval scores or native profiles differ")
	}
	var native struct{ Profile, Selection, Scores artifact.ID }
	read(parity.NativeScores, &native)
	if parity.AcquisitionProfile != native.Profile || parity.RawSelection != native.Selection || parity.ScoreRows != native.Scores {
		t.Fatal("IFEval parity substituted native inputs")
	}
	var scores struct{ Metrics map[string]float64 }
	read(native.Scores, &scores)
	if !maps.Equal(parity.Metrics, scores.Metrics) {
		t.Fatal("IFEval parity changed retained metrics")
	}
	finalized, found, err := runrecord.GateFinalizationForPreparation(t.Context(), store, parity.GatePreparation)
	if err != nil || !found || finalized.ID != parity.GateFinalization || finalized.CodeCommit != parity.ScorerCommit || finalized.Outcome != runrecord.OutcomeSucceeded || finalized.Result == nil || *finalized.Result != parity.GateResult {
		t.Fatalf("IFEval scorer gate is not uniquely finalized: %v", err)
	}
	gate, err := runrecord.RequireGateResult(t.Context(), store, parity.GateResult)
	if err != nil || gate.CodeCommit != parity.ScorerCommit || gate.Outcome != runrecord.OutcomeSucceeded {
		t.Fatalf("IFEval scorer gate failed or changed: %v", err)
	}
	accepted := false
	for _, step := range gate.Steps {
		if step.Name == "acceptance" && (step.Outcome == runrecord.StepSucceeded || step.Outcome == runrecord.StepReused) {
			accepted = true
		}
	}
	if !accepted || !strings.Contains(parity.Verify, "TestIFEvalRetainedTextAcceptance") || !strings.Contains(parity.Verify, "TestIFEvalLanguageAcceptance") {
		t.Fatal("IFEval complete parity acceptance is absent")
	}
	t.Log("541 prompts, 834 instructions, 1668 exact native/Go verdicts; scores and committed acceptance reused, no model executions")
}
