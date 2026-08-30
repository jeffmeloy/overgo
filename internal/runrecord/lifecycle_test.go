package runrecord

import (
	"context"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func TestGateLifecycleAndDebt(t *testing.T) {
	environment, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("lifecycle-environment"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("lifecycle-result"))
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := NewGatePreparation(string(make([]byte, 64)), environment, time.Unix(100, 0))
	if err == nil {
		t.Fatal("accepted a non-hex tree key")
	}
	prepared, err = NewGatePreparation("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", environment, time.Unix(100, 0))
	if err != nil {
		t.Fatal(err)
	}
	debt, err := OutstandingGateDebt([]GateLifecycle{prepared})
	if err != nil || len(debt) != 1 || debt[0].ID != prepared.ID {
		t.Fatalf("prepared debt = (%+v, %v)", debt, err)
	}
	finalized, err := NewGateFinalization(prepared, "0123456789abcdef0123456789abcdef01234567", result, OutcomeSucceeded)
	if err != nil {
		t.Fatal(err)
	}
	debt, err = OutstandingGateDebt([]GateLifecycle{prepared, finalized})
	if err != nil || len(debt) != 0 {
		t.Fatalf("finalized debt = (%+v, %v)", debt, err)
	}
}

func TestGateLifecycleStoreScanAndDebt(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if GateLifecycleCurrentAlias != overgodb.StoreLocalAliasPrefix+"gate-lifecycle/current" {
		t.Fatalf("gate lifecycle alias = %q", GateLifecycleCurrentAlias)
	}
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "store-lifecycle-environment")
	result := testutil.ArtifactID(t, artifact.KindEvidence, "store-lifecycle-result")
	testutil.PublishArtifact(t, store, environment)
	prepared, err := NewGatePreparation(
		"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		environment, time.Unix(100, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	preparedContent, err := prepared.Content()
	if err != nil {
		t.Fatal(err)
	}
	unrelatedContract := artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: "application/json", Schema: "test/unrelated-lifecycle/v1",
	}
	unrelated, err := unrelatedContract.ContentBytes([]byte(`{"state":"prepared"}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "test/gate-lifecycle/prepared", Contents: []artifact.Content{preparedContent, unrelated},
		Lineage: prepared.Lineage(),
		Aliases: []artifact.AliasBinding{{Name: GateLifecycleCurrentAlias, Target: prepared.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	records, err := GateLifecyclesInStore(ctx, store)
	if err != nil || len(records) != 1 || records[0].ID != prepared.ID {
		t.Fatalf("stored prepared records = (%+v, %v)", records, err)
	}
	debt, err := OutstandingGateDebt(records)
	if err != nil || len(debt) != 1 || debt[0].ID != prepared.ID {
		t.Fatalf("stored prepared debt = (%+v, %v)", debt, err)
	}

	testutil.PublishArtifact(t, store, result)
	finalized, err := NewGateFinalization(
		prepared, "0123456789abcdef0123456789abcdef01234567", result, OutcomeSucceeded,
	)
	if err != nil {
		t.Fatal(err)
	}
	finalizedContent, err := finalized.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "test/gate-lifecycle/finalized", Contents: []artifact.Content{finalizedContent},
		Lineage: finalized.Lineage(),
		Aliases: []artifact.AliasBinding{{
			Name: GateLifecycleCurrentAlias, Target: finalized.ID, Previous: artifact.IDPointer(prepared.ID),
		}},
	}); err != nil {
		t.Fatal(err)
	}
	records, err = GateLifecyclesInStore(ctx, store)
	if err != nil || len(records) != 2 || records[0].ID != prepared.ID || records[1].ID != finalized.ID {
		t.Fatalf("stored finalized records = (%+v, %v)", records, err)
	}
	debt, err = OutstandingGateDebt(records)
	if err != nil || len(debt) != 0 {
		t.Fatalf("stored finalized debt = (%+v, %v)", debt, err)
	}
	if _, err := GateLifecyclesInStore(nil, store); err == nil {
		t.Fatal("nil lifecycle context accepted")
	}
	if _, err := GateLifecyclesInStore(ctx, nil); err == nil {
		t.Fatal("nil lifecycle store accepted")
	}
}

func TestGateHeartbeat(t *testing.T) {
	now := time.Unix(100, 0)
	heartbeat := GateHeartbeat{State: HeartbeatRunning, Updated: now}
	if got := heartbeat.Watchdog(now.Add(4*time.Second), 5*time.Second); got != HeartbeatRunning {
		t.Fatalf("fresh heartbeat = %s", got)
	}
	if got := heartbeat.Watchdog(now.Add(6*time.Second), 5*time.Second); got != HeartbeatStale {
		t.Fatalf("old heartbeat = %s", got)
	}
	heartbeat.State = HeartbeatRecordDebt
	if got := heartbeat.Watchdog(now.Add(time.Hour), 5*time.Second); got != HeartbeatRecordDebt {
		t.Fatalf("terminal heartbeat = %s", got)
	}
}
