package agenttool

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

func TestStableCapabilityProxy(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	target := testcheck.Must(NewManual(Manual{Name: "echo.value", Description: "Return exact arguments.", Effect: EffectInspection,
		Arguments: []Field{{Name: "value", Kind: FieldString, Required: true}}, Transport: Transport{Kind: TransportBuiltin}}))(t)
	if _, err := PublishManualCatalog(ctx, store, []Manual{target}); err != nil {
		t.Fatal(err)
	}
	publishActiveProxyCatalog(t, store, target)
	executor := NewOperatorExecutor()
	testcheck.NoError(t, executor.registerBuiltin(target.Name, func(_ context.Context, raw json.RawMessage) (json.RawMessage, error) { return raw, nil }))
	admitted := artifact.ID{}
	testcheck.NoError(t, RegisterCapabilityProxy(executor, store, func(effect InvocationEffect) error { admitted = effect.Manual; return nil }))
	proxy := testcheck.Must(CapabilityProxyManual())(t)
	request, _ := json.Marshal(map[string]any{"action": "call", "manual": target.ID.String(), "arguments": map[string]any{"value": "ok"}})
	result := testcheck.Must(executor.Invoke(ctx, proxy, request))(t)
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
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	target := testcheck.Must(NewManual(Manual{
		Name: "policy.probe", Description: "Exercise current argv authority.", Effect: EffectInspection,
		Transport: Transport{Kind: TransportArgv, Program: "overgo-probe"},
	}))(t)
	if _, err := PublishArgvPolicy(ctx, store, []string{"overgo-probe"}); err != nil {
		t.Fatal(err)
	}
	if _, err := PublishManualCatalog(ctx, store, []Manual{target}); err != nil {
		t.Fatal(err)
	}
	publishActiveProxyCatalog(t, store, target)
	executor := NewOperatorExecutor()
	admitted := false
	testcheck.NoError(t, RegisterCapabilityProxy(executor, store, func(InvocationEffect) error { admitted = true; return nil }))
	if _, err := PublishArgvPolicy(ctx, store, nil); err != nil {
		t.Fatal(err)
	}
	proxy := testcheck.Must(CapabilityProxyManual())(t)
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
	snapshot := testcheck.Must(NewCatalogSnapshot(manuals))(t)
	content := testcheck.Must(snapshot.ArtifactContent())(t)
	batch := testcheck.Must(artifact.NewDocumentBatch("proxy-test/"+snapshot.ID.String(), []artifact.Content{content}, nil,
		[]artifact.AliasBinding{{Name: ActiveCatalogAlias, Target: snapshot.ID}}))(t)
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
}
