package agenttool

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

func TestStableCapabilityProxy(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	target, err := NewManual(Manual{Name: "echo.value", Description: "Return exact arguments.", Effect: EffectInspection,
		Arguments: []Field{{Name: "value", Kind: FieldString, Required: true}}, Transport: Transport{Kind: TransportBuiltin}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PublishManualCatalog(ctx, store, []Manual{target}); err != nil {
		t.Fatal(err)
	}
	publishActiveProxyCatalog(t, store, target)
	executor := NewOperatorExecutor()
	if err := executor.registerBuiltin(target.Name, func(_ context.Context, raw json.RawMessage) (json.RawMessage, error) { return raw, nil }); err != nil {
		t.Fatal(err)
	}
	admitted := artifact.ID{}
	if err := RegisterCapabilityProxy(executor, store, func(effect InvocationEffect) error { admitted = effect.Manual; return nil }); err != nil {
		t.Fatal(err)
	}
	proxy, err := CapabilityProxyManual()
	if err != nil {
		t.Fatal(err)
	}
	request, _ := json.Marshal(map[string]any{"action": "call", "manual": target.ID.String(), "arguments": map[string]any{"value": "ok"}})
	result, err := executor.Invoke(ctx, proxy, request)
	if err != nil {
		t.Fatal(err)
	}
	if admitted != target.ID || string(result) != `{"value":"ok"}` {
		t.Fatalf("proxy result=%s admitted=%s", result, admitted)
	}
	other, _ := NewManual(Manual{Name: "other.tool", Description: "Unlisted.", Effect: EffectInspection, Transport: Transport{Kind: TransportBuiltin}})
	request, _ = json.Marshal(map[string]any{"action": "inspect", "manual": other.ID.String()})
	if _, err := executor.Invoke(ctx, proxy, request); err == nil {
		t.Fatal("proxy admitted a manual outside active snapshot")
	}
}

func TestCapabilityProxyRechecksArgvAuthority(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	target, err := NewManual(Manual{
		Name: "policy.probe", Description: "Exercise current argv authority.", Effect: EffectInspection,
		Transport: Transport{Kind: TransportArgv, Program: "overgo-probe"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PublishArgvPolicy(ctx, store, []string{"overgo-probe"}); err != nil {
		t.Fatal(err)
	}
	if _, err := PublishManualCatalog(ctx, store, []Manual{target}); err != nil {
		t.Fatal(err)
	}
	publishActiveProxyCatalog(t, store, target)
	executor := NewOperatorExecutor()
	admitted := false
	if err := RegisterCapabilityProxy(executor, store, func(InvocationEffect) error { admitted = true; return nil }); err != nil {
		t.Fatal(err)
	}
	if _, err := PublishArgvPolicy(ctx, store, nil); err != nil {
		t.Fatal(err)
	}
	proxy, err := CapabilityProxyManual()
	if err != nil {
		t.Fatal(err)
	}
	request, _ := json.Marshal(map[string]any{"action": "call", "manual": target.ID.String(), "arguments": map[string]any{}})
	if _, err := executor.Invoke(ctx, proxy, request); err == nil || !strings.Contains(err.Error(), "outside the committed policy") {
		t.Fatalf("tightened argv policy result = %v", err)
	}
	if admitted {
		t.Fatal("capability proxy reached effect admission after argv authority refused")
	}
}

func publishActiveProxyCatalog(t *testing.T, store artifact.Repository, manuals ...Manual) {
	t.Helper()
	snapshot, err := NewCatalogSnapshot(manuals)
	if err != nil {
		t.Fatal(err)
	}
	content, err := snapshot.ArtifactContent()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch("proxy-test/"+snapshot.ID.String(), []artifact.Content{content}, nil,
		[]artifact.AliasBinding{{Name: ActiveCatalogAlias, Target: snapshot.ID}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(context.Background(), store, batch); err != nil {
		t.Fatal(err)
	}
}
