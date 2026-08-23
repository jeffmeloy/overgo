package main

import (
	"bytes"
	"context"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/finding"
	"overgo/internal/repodb"
)

func TestRecordFindingRequiresVerifierAndUsesTypedStore(t *testing.T) {
	repository := filepath.Join(t.TempDir(), "store")
	base := []string{
		"-repo", repository, "-title", "stale ranking document", "-severity", "high",
		"-owner", "docs/report.md", "-evidence", "report ranks a completed gate defect",
		"-closure", "bind the report to generated evidence",
	}
	if err := runArgs(base, &bytes.Buffer{}); err == nil {
		t.Fatal("finding without failable check was recorded")
	}
	var output bytes.Buffer
	if err := runArgs(append(base, "-check", "go test ./cmd/gate -run '^TestDocFreshness$'"), &output); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output.String(), "FINDING parked: evidence:sha256:") {
		t.Fatalf("output = %q", output.String())
	}
	store, err := repodb.Open(repository)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	result, err := store.Query(context.Background(), repodb.Query{
		Kind: artifact.KindEvidence, MaxResults: 100, Projection: repodb.ProjectArtifacts,
	})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, descriptor := range result.Artifacts {
		if descriptor.MediaType != finding.MediaType {
			continue
		}
		content, ok, err := store.Content(context.Background(), descriptor.ID)
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
