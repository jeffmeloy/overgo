package agenttool

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestCapabilityIdentityContract(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	implementation := testutil.ArtifactID(t, artifact.KindFile, "tool-capability-implementation")
	schema := testutil.ArtifactID(t, artifact.KindProfile, "tool-capability-schema")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "tool/capability/authority", Artifacts: []artifact.Descriptor{{ID: implementation}, {ID: schema}}}); err != nil {
		t.Fatal(err)
	}
	identity := runrecord.CapabilityIdentity{
		Implementation: implementation, Release: "2.0.1",
		Transport: runrecord.CapabilityTransport{Kind: runrecord.CapabilityTransportHTTP, Endpoint: "https://tool.example/run", Protocol: "strict-json/1"},
		Schema:    schema, Platform: runrecord.CapabilityPlatform{OS: "linux", Arch: "amd64"},
		Resources: runrecord.CapabilityResourceEnvelope{
			MaxInputBytes: 4096, MaxOutputBytes: 8192, MaxConcurrent: 2, CPUThreads: 2, HostBytes: 1 << 30,
		},
	}
	manual, err := NewManual(Manual{
		Name: "capability.inspect", Description: "Inspect through one exact capability.", Effect: EffectInspection,
		Transport: Transport{Kind: TransportHTTP, URL: "https://tool.example/run", Protocol: "strict-json/1"}, CapabilityIdentity: &identity,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PublishManualCatalog(ctx, store, []Manual{manual}); err != nil {
		t.Fatal(err)
	}
	resolved, err := ResolveRegisteredManual(ctx, store, manual.Name)
	if err != nil || resolved.Capability != manual.Capability {
		t.Fatalf("resolved manual = (%+v, %v)", resolved, err)
	}
	capability, err := runrecord.RequireCapabilityIdentity(ctx, store, resolved.Capability)
	if err != nil || capability.Implementation != implementation || capability.Schema != schema {
		t.Fatalf("resolved capability = (%+v, %v)", capability, err)
	}
	mismatch := identity
	mismatch.Transport.Endpoint = "https://tool.example/other"
	if _, err := NewManual(Manual{
		Name: "capability.mismatch", Description: "Refuse a transport mismatch.", Effect: EffectInspection,
		Transport: Transport{Kind: TransportHTTP, URL: "https://tool.example/run", Protocol: "strict-json/1"}, CapabilityIdentity: &mismatch,
	}); err == nil {
		t.Fatal("manual admitted a different same-kind capability endpoint")
	}
	if manualTransportMatchesCapability(
		Transport{Kind: TransportArgv, Program: "runner", Args: []string{"serve"}, Protocol: "stdin-json/1"},
		runrecord.CapabilityTransport{Kind: runrecord.CapabilityTransportArgv, Endpoint: "runner", Args: []string{"other"}, Protocol: "stdin-json/1"},
	) {
		t.Fatal("manual admitted different argv authority words")
	}
	if manualTransportMatchesCapability(
		Transport{Kind: TransportMCPHTTP, URL: "https://mcp.example", Target: "inspect", Protocol: "mcp/1"},
		runrecord.CapabilityTransport{Kind: runrecord.CapabilityTransportMCPHTTP, Endpoint: "https://mcp.example", Target: "mutate", Protocol: "mcp/1"},
	) {
		t.Fatal("manual admitted a different MCP target")
	}
	protocolDrift := identity
	protocolDrift.Transport.Protocol = "strict-json/2"
	if _, err := NewManual(Manual{
		Name: "capability.protocol", Description: "Refuse protocol drift.", Effect: EffectInspection,
		Transport: Transport{Kind: TransportHTTP, URL: "https://tool.example/run", Protocol: "strict-json/1"}, CapabilityIdentity: &protocolDrift,
	}); err == nil {
		t.Fatal("manual admitted a different capability protocol")
	}
	driftedAuthority := testutil.ArtifactID(t, artifact.KindProfile, "different-capability")
	if _, err := NewManual(Manual{
		Name: "capability.drift", Description: "Refuse supplied authority drift.", Effect: EffectInspection,
		Transport: Transport{Kind: TransportHTTP, URL: "https://tool.example/run", Protocol: "strict-json/1"}, Capability: driftedAuthority, CapabilityIdentity: &identity,
	}); err == nil {
		t.Fatal("manual silently replaced a supplied capability authority")
	}
}

func TestExactCapabilityPlacementAndReceipt(t *testing.T) {
	TestCapabilityIdentityContract(t)
}
