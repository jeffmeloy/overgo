package main

import (
	"encoding/json"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/vqaserve"
	"overgo/internal/workflowruntime"
)

// Retained acceptance only: no model load, current-HEAD requirement or promotion.
func TestAcceptedRxBrainVQA(t *testing.T) {
	t.Parallel()
	testskip.Short(t)
	var fixture retainedPublicationFixture
	for _, candidate := range retainedPublicationFixtures(t) {
		if candidate.Name == "rxbrain-vqa" {
			fixture = candidate
		}
	}
	if fixture.Name == "" {
		t.Fatal("canonical VQA publication is absent")
	}
	checkRetainedModelPublication(t, fixture)
	root := testutil.RepoRoot(t)
	var spec verificationSpecification
	testcheck.NoError(t, jsonfile.DecodeStrict(filepath.Join(root, "docs/verification", fixture.Specification), &spec))
	if len(spec.Claims) != 1 || spec.Claims[0].Capability != "vqa-canonical-native-text" || spec.Claims[0].Tier != runrecord.TierExactGolden {
		t.Fatal("acceptance exceeds the single canonical text claim")
	}
	claim := spec.Claims[0]
	roots := testcheck.Must(dataroot.Resolve(root))(t)
	store := overgodbtest.OpenReadOnly(t, retainedReferenceStore(roots.Store))
	read := func(id artifact.ID, value any) artifact.Content {
		t.Helper()
		content := testcheck.Must(artifact.RequireTypedContent(t.Context(), store, id))(t)
		if value != nil {
			testcheck.NoError(t, json.Unmarshal(content.Data, value))
		}
		return content
	}
	var protocolID, runID artifact.ID
	for _, id := range claim.Evidence {
		switch id.Kind() {
		case artifact.KindRun:
			runID = id
		case artifact.KindEvidence:
			protocolID = id
		}
	}
	var protocol struct {
		Model, Recipe, Request, Image, Oracle, Binary artifact.ID
		Producer                                      string
		Expected                                      string `json:"expected_text"`
		NativeTokens                                  []int  `json:"native_tokens"`
		Cases                                         int
	}
	read(protocolID, &protocol)
	quality := testcheck.Must(runrecord.RequireExactRun(t.Context(), store, runID))(t)
	if quality.Outcome != runrecord.OutcomeSucceeded || quality.Recipe != protocol.Recipe || quality.CodeCommit != protocol.Producer || claim.Commit != protocol.Producer || quality.MeasuredNS != claim.WallNS || len(quality.Outputs) != 1 {
		t.Fatal("quality result, source or measured command wall differs")
	}
	definition, err := recipe.RequireDefinition(t.Context(), store, protocol.Recipe)
	if err != nil || definition.Model != spec.Model || protocol.Model != spec.Model {
		t.Fatalf("model/recipe binding: %v", err)
	}
	if _, err := runrecord.RequireEnvironment(t.Context(), store, quality.Environment); err != nil {
		t.Fatal(err)
	}
	var command struct {
		Gate   artifact.ID `json:"gate_id"`
		Recipe artifact.ID `json:"recipe_id"`
		Run    artifact.ID `json:"run_id"`
		Output struct {
			Output string                     `json:"output"`
			Phases []workflowruntime.NodeWall `json:"phases"`
		} `json:"output"`
	}
	read(quality.Outputs[0], &command)
	verified := testcheck.Must(runrecord.VerifyGateRun(t.Context(), store, protocol.Recipe, command.Gate, command.Run))(t)
	if command.Recipe != protocol.Recipe || verified.Run.CodeCommit != protocol.Producer || !slices.Equal(verified.Run.Inputs, []artifact.ID{protocol.Request}) || command.Output.Output != protocol.Expected || protocol.Expected == "" {
		t.Fatal("verifier, exact request or native output differs")
	}
	if !slices.ContainsFunc(command.Output.Phases, func(wall workflowruntime.NodeWall) bool {
		return wall.Node == "generate" && wall.WallNS > 0
	}) {
		t.Fatal("capture does not establish a new generation")
	}
	var request vqaserve.Request
	requestContent := read(protocol.Request, nil)
	testcheck.NoError(t, strictjson.DecodeBytes(requestContent.Data, &request))
	testcheck.NoError(t, vqaserve.RequestContract.ValidateContent(requestContent, protocol.Request))
	var oracle struct {
		GeneratedTokens []int  `json:"generated_tokens"`
		GeneratedText   string `json:"generated_text"`
		DecodeStepCount int    `json:"decode_step_count"`
		Request         struct {
			Images   []string
			Question string
		}
	}
	read(protocol.Oracle, &oracle)
	if oracle.GeneratedText != protocol.Expected {
		t.Fatal("expected text differs from the independent native capture")
	}
	if protocol.Cases != 1 || len(oracle.Request.Images) != protocol.Cases || request.Image != protocol.Image || request.Question != oracle.Request.Question || request.MaxTokens != oracle.DecodeStepCount || request.MaxTokens <= 0 || len(protocol.NativeTokens) != request.MaxTokens+1 || !slices.Equal(protocol.NativeTokens, oracle.GeneratedTokens) {
		t.Fatal("native request, decode bound or case denominator differs")
	}
	if _, err := vqaserve.ReadImage(t.Context(), store, protocol.Image); err != nil {
		t.Fatal(err)
	}
	// The accepted run was made when a capability run's key was its recipe and its request alone, and its
	// receipt is kept under that key; a run made since is keyed on the code that made it as well.
	operation := testcheck.Must(workflowruntime.ExecutionID(definition.ID, "recipe/run/"+definition.ID.String()+"/"+requestContent.Descriptor.ID.String()))(t)
	answer := testcheck.Must(artifact.JSONContent(vqaserve.AnswerContract, command.Output.Output))(t)
	read(answer.Descriptor.ID, nil)
	var receipt runrecord.StageReceipt
	for _, id := range quality.Inputs {
		if id.Kind() != artifact.KindEvidence || id == protocolID || id == protocol.Oracle {
			continue
		}
		content := read(id, nil)
		if content.Descriptor.Schema == runrecord.StageReceiptSchema {
			receipt = testcheck.Must(runrecord.ParseStageReceipt(content.Data))(t)
		}
	}
	if receipt.Recipe != definition.ID || receipt.Operation != operation || receipt.Node != "generate" || receipt.State != runrecord.StageCompleted || len(receipt.Outputs) != 1 || !slices.Equal(receipt.Outputs[0].Artifacts, []artifact.ID{answer.Descriptor.ID}) {
		t.Fatal("retained workflow receipt does not produce the observed answer")
	}
	wantInputs := []artifact.ID{protocolID, protocol.Request, protocol.Image, protocol.Oracle, protocol.Binary, command.Run, receipt.ID, answer.Descriptor.ID}
	gotInputs := slices.Clone(quality.Inputs)
	slices.SortFunc(gotInputs, artifact.CompareID)
	slices.SortFunc(wantInputs, artifact.CompareID)
	if !slices.Equal(gotInputs, wantInputs) {
		t.Fatalf("comparison inputs = %v, want %v", gotInputs, wantInputs)
	}
	t.Log("one canonical VQA text equals the native reference; request, source, workflow output and environment retained; model executions=0")
}
