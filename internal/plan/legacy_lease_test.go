package plan

import (
	"encoding/json"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

func TestRetireLegacyLeases(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()

	// One live lease under the current contract.
	leaseData, _ := json.Marshal(WorkLease{
		Version: workLeaseVersion, Task: "task", Worktree: "C:/worktree", Branch: "codex/task", Role: "developer",
		TargetHead: "0123456789abcdef0123456789abcdef01234567", ConflictsWith: []string{},
		Resources: Resources{CPUThreads: 8, HostRAMGiB: 16, VRAMGiB: 8},
		ExpiresAt: time.Now().Add(time.Hour).UTC().Format(time.RFC3339Nano),
	})
	live, err := RecordWorkLease(ctx, store, leaseData)
	if err != nil {
		t.Fatal(err)
	}

	// One pre-contract document under the lease alias prefix.
	contract := artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: WorkLeaseMediaType, Schema: WorkLeaseSchema,
	}
	legacyData := []byte(`{"version":0,"task":"legacy"}`)
	legacyID, err := contract.Identify(legacyData)
	if err != nil {
		t.Fatal(err)
	}
	legacyContent, err := contract.Content(legacyID, legacyData)
	if err != nil {
		t.Fatal(err)
	}
	legacyAlias := WorkLeaseAliasRoot + "legacy"
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key: "test/legacy-lease", Contents: []artifact.Content{legacyContent},
		Aliases: []artifact.AliasBinding{{Name: legacyAlias, Target: legacyID}},
	}); err != nil {
		t.Fatal(err)
	}

	if _, err := RetireLegacyLeases(ctx, store, 2); err == nil {
		t.Fatal("mismatched reviewed expectation was accepted")
	}
	retired, err := RetireLegacyLeases(ctx, store, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(retired) != 1 || retired[0] != legacyAlias {
		t.Fatalf("retired = %v", retired)
	}
	if _, found, err := artifact.ResolveAlias(ctx, store, legacyAlias); err != nil || found {
		t.Fatalf("legacy alias survived retirement: found=%t err=%v", found, err)
	}
	parsed, ok, err := ReadWorkLease(ctx, store, live.ID)
	if err != nil || !ok || parsed.Task != "task" {
		t.Fatalf("live lease = (%+v, %t, %v)", parsed, ok, err)
	}
	again, err := RetireLegacyLeases(ctx, store, 0)
	if err != nil || len(again) != 0 {
		t.Fatalf("second retirement = (%v, %v)", again, err)
	}
}
