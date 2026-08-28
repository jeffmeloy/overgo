package capabilityruntime

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestCapabilityIdentityContract(t *testing.T) {
	fixture := runrecord.CapabilityIdentity{
		Implementation: testutil.ArtifactID(t, artifact.KindFile, "capability-executable"),
		Release:        "1.4.2",
		Transport:      runrecord.CapabilityTransport{Kind: runrecord.CapabilityTransportHTTP, Endpoint: "https://worker.example/v1/run", Protocol: "overgo-rpc/3"},
		Schema:         testutil.ArtifactID(t, artifact.KindProfile, "capability-schema"),
		Platform:       runrecord.CapabilityPlatform{OS: "linux", Arch: "amd64", Accelerator: "cuda-sm90"},
		Resources: runrecord.CapabilityResourceEnvelope{
			MaxInputBytes: 1 << 20, MaxOutputBytes: 2 << 20, MaxConcurrent: 4,
			CPUThreads: 8, HostBytes: 16 << 30, DeviceBytes: 24 << 30,
		},
	}
	first, err := fixture.Identify()
	if err != nil || first.ID.Kind() != artifact.KindProfile || first.ValidateIdentity() != nil {
		t.Fatalf("capability identity = (%+v, %v)", first, err)
	}
	second, err := fixture.Identify()
	if err != nil || second.ID != first.ID {
		t.Fatalf("stable identity = (%s, %v), want %s", second.ID, err, first.ID)
	}
	mutations := []func(*runrecord.CapabilityIdentity){
		func(value *runrecord.CapabilityIdentity) { value.Release = "1.4.3" },
		func(value *runrecord.CapabilityIdentity) { value.Transport.Protocol = "overgo-rpc/4" },
		func(value *runrecord.CapabilityIdentity) {
			value.Schema = testutil.ArtifactID(t, artifact.KindProfile, "other-schema")
		},
		func(value *runrecord.CapabilityIdentity) { value.Platform.Arch = "arm64" },
		func(value *runrecord.CapabilityIdentity) { value.Resources.MaxConcurrent++ },
	}
	for index, mutate := range mutations {
		candidate := fixture
		mutate(&candidate)
		identified, identifyErr := candidate.Identify()
		if identifyErr != nil || identified.ID == first.ID {
			t.Fatalf("mutation %d identity = (%s, %v)", index, identified.ID, identifyErr)
		}
	}
}
