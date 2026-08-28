package agentloop

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/operatoraction"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestAgentMutationCheckpointLifecycle(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	target := filepath.Join(root, "state.txt")
	if err := os.WriteFile(target, []byte("before"), 0o640); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	manual, err := agenttool.NewManual(agenttool.Manual{Name: "file.write", Description: "Write one file.", Effect: agenttool.EffectMutation,
		Arguments: []agenttool.Field{{Name: "path", Kind: agenttool.FieldString, Required: true}}, Ceiling: agenttool.EffectCeiling{Targets: []agenttool.EffectTargetBinding{{Scope: agenttool.EffectScopeWorkspace, Argument: "path"}}}, Transport: agenttool.Transport{Kind: agenttool.TransportBuiltin}})
	if err != nil {
		t.Fatal(err)
	}
	arguments, _ := json.Marshal(map[string]string{"path": target})
	effect, err := agenttool.DeriveInvocationEffect(manual, arguments, nil)
	if err != nil {
		t.Fatal(err)
	}
	runtime, err := NewMutationCheckpointRuntime(root, store)
	if err != nil {
		t.Fatal(err)
	}
	checkpoint, err := runtime.BeginCheckpoint(ctx, testutil.ArtifactID(t, artifact.KindEvidence, "operation"), effect, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, []byte("after"), 0o640); err != nil {
		t.Fatal(err)
	}
	checkpoint, err = runtime.SealCheckpoint(ctx, checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "recipe")
	request, err := RestoreApprovalRequest(checkpoint, recipeID)
	if err != nil {
		t.Fatal(err)
	}
	decision, err := runrecord.NewHumanDecision(request, operatoraction.AnswerGrant)
	if err != nil {
		t.Fatal(err)
	}
	if err := runrecord.PublishHumanDecision(ctx, store, request, decision); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, []byte("drift"), 0o640); err != nil {
		t.Fatal(err)
	}
	if _, err := runtime.RestoreCheckpoint(ctx, checkpoint, effect, recipeID); err == nil {
		t.Fatal("postimage drift was accepted")
	}
	if err := os.WriteFile(target, []byte("after"), 0o640); err != nil {
		t.Fatal(err)
	}
	receipt, err := runtime.RestoreCheckpoint(ctx, checkpoint, effect, recipeID)
	if err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(target)
	if string(data) != "before" || !receipt.Valid() {
		t.Fatalf("restored %q receipt=%s", data, receipt)
	}
}
