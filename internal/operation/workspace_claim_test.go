package operation

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestWorkspaceClaimLifecycle(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	leaseJSON, _ := json.Marshal(plan.WorkLease{Version: 1, Task: "claim/test", Worktree: "C:/repo/claim", Branch: "codex/claim", Role: "developer",
		TargetHead: "0123456789abcdef0123456789abcdef01234567", ConflictsWith: []string{}, Resources: plan.Resources{CPUThreads: 1, HostRAMGiB: 1},
		Claims: plan.WorkspaceClaims{Write: []string{"C:/repo/claim/internal"}}, ExpiresAt: time.Now().Add(time.Hour).UTC().Format(time.RFC3339Nano)})
	lease, err := plan.RecordWorkLease(ctx, store, leaseJSON)
	if err != nil {
		t.Fatal(err)
	}
	manual, _ := agenttool.NewManual(agenttool.Manual{Name: "repo.write", Description: "Write repository.", Effect: agenttool.EffectMutation,
		Arguments: []agenttool.Field{{Name: "path", Kind: agenttool.FieldString, Required: true}}, Ceiling: agenttool.EffectCeiling{Targets: []agenttool.EffectTargetBinding{{Scope: agenttool.EffectScopeWorkspace, Argument: "path"}}}, Transport: agenttool.Transport{Kind: agenttool.TransportBuiltin}})
	effect, err := agenttool.DeriveInvocationEffect(manual, json.RawMessage(`{"path":"C:/repo/claim/internal/file.go"}`), nil)
	if err != nil {
		t.Fatal(err)
	}
	manager, err := NewManagerWithRepository(4, store)
	if err != nil {
		t.Fatal(err)
	}
	defer manager.Close()
	request := Request{Task: recipe.TaskInference, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "recipe"), Lease: &lease, Effect: &effect}
	id, err := manager.Submit(ctx, request, func(context.Context, Reporter) (Completion, error) {
		return Completion{Run: testutil.ArtifactID(t, artifact.KindRun, "run")}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	status, err := manager.Wait(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	if status.WorkspaceClaim == nil || status.WorkspaceClaim.State != WorkspaceClaimReleased || status.WorkspaceClaim.Previous == nil || !status.WorkspaceClaim.Previous.Valid() {
		t.Fatalf("claim lifecycle = %+v", status.WorkspaceClaim)
	}
}
