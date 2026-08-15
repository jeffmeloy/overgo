package plan

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestRoadmapProbeBindsExpectedFailure(t *testing.T) {
	probe, err := roadmapProbeCodec.New(RoadmapProbe{
		Version: roadmapProbeVersion, Row: "refusal-ledger",
		Verifier:        "go test ./internal/runrecord -run '^TestRefusalDecisionCarriesMeasuredReason$' -count=1 -v",
		CodeCommit:      "0123456789abcdef0123456789abcdef01234567",
		Run:             testutil.ArtifactID(t, artifact.KindRun, "isolated-failed-probe"),
		ExpectedFailure: "capability-absent",
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := roadmapProbeCodec.Content(probe)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := roadmapProbeCodec.Parse(content.Data)
	if err != nil || parsed.ID != probe.ID || parsed.Row != probe.Row {
		t.Fatalf("probe round trip = (%+v, %v)", parsed, err)
	}
	probe.Verifier = "go test ./internal/runrecord"
	if _, err := roadmapProbeCodec.New(probe); err == nil {
		t.Fatal("unnamed probe verifier passed")
	}
}
