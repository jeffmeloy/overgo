package main

import (
	"bytes"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/finding"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

func TestRecordFindingRequiresVerifierAndUsesTypedStore(t *testing.T) {
	repository := t.TempDir()
	base := []string{
		"-repo", repository, "-title", "stale ranking document", "-severity", "high",
		"-owner", "docs/report.md", "-evidence", "report ranks a completed gate defect",
		"-closure", "bind the report to generated evidence",
	}
	if err := runArgs(base, &bytes.Buffer{}); err == nil {
		t.Fatal("finding without failable check was recorded")
	}
	var output bytes.Buffer
	testcheck.NoError(t, runArgs(append(base, "-check", "go test ./cmd/gate -run '^TestDocFreshness$'"), &output))
	if !strings.Contains(output.String(), "FINDING parked: evidence:sha256:") {
		t.Fatalf("output = %q", output.String())
	}
	store := overgodbtest.Create(t, repository)
	result, err := store.Query(t.Context(), overgodb.Query{
		Kind: artifact.KindEvidence, MaxResults: 100, Projection: overgodb.ProjectArtifacts,
	})
	testcheck.NoError(t, err)
	found := false
	for _, descriptor := range result.Artifacts {
		if descriptor.MediaType != finding.MediaType {
			continue
		}
		content, ok, err := artifact.ReadContent(t.Context(), store, descriptor.ID)
		if err != nil || !ok {
			t.Fatalf("finding content: ok=%v err=%v", ok, err)
		}
		document, err := finding.Parse(content.Data)
		if err != nil || document.FailableCheck == "" || len(document.Lineage()) != 2 {
			t.Fatalf("typed finding = %+v err=%v", document, err)
		}
		found = true
	}
	if !found {
		t.Fatal("typed finding not stored")
	}
}
