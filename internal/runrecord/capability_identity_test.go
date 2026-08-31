package runrecord

import (
	"reflect"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func TestCapabilityIdentityContract(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	implementation := testutil.ArtifactID(t, artifact.KindFile, "capability-implementation")
	schema := testutil.ArtifactID(t, artifact.KindProfile, "capability-schema")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "capability/identity/parents", Artifacts: []artifact.Descriptor{{ID: implementation}, {ID: schema}},
	}); err != nil {
		t.Fatal(err)
	}
	identity, err := (CapabilityIdentity{
		Implementation: implementation,
		Release:        "2.0.1",
		Transport:      CapabilityTransport{Kind: CapabilityTransportHTTP, Endpoint: "https://tool.example/run", Protocol: "strict-json/1"},
		Schema:         schema,
		Platform:       CapabilityPlatform{OS: "linux", Arch: "amd64"},
		Resources: CapabilityResourceEnvelope{
			MaxInputBytes: 4096, MaxOutputBytes: 8192, MaxConcurrent: 2, CPUThreads: 2, HostBytes: 1 << 30,
		},
	}).Identify()
	if err != nil {
		t.Fatal(err)
	}
	content, err := identity.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch("capability/identity", []artifact.Content{content}, identity.Lineage(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	resolved, err := RequireCapabilityIdentity(ctx, store, identity.ID)
	if err != nil || !reflect.DeepEqual(resolved, identity) {
		t.Fatalf("resolved capability = (%+v, %v), want %+v", resolved, err, identity)
	}

	missingProtocol := identity
	missingProtocol.ID = artifact.ID{}
	missingProtocol.Transport.Protocol = ""
	if _, err := missingProtocol.Identify(); err == nil {
		t.Fatal("capability admitted a missing protocol revision")
	}
	deviceDrift := identity
	deviceDrift.ID = artifact.ID{}
	deviceDrift.Resources.DeviceBytes = 1024
	if _, err := deviceDrift.Identify(); err == nil {
		t.Fatal("capability admitted device resources without an accelerator")
	}
}
