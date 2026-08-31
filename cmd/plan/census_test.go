package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/closurescan"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
)

func TestPublishedCensusBaseline(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	document := plan.Plan{Campaign: "campaign", Doctrine: "Measured baselines live in OvergoDB.", Items: []plan.Item{}}
	if err := plan.Save(filepath.Join(root, filepath.FromSlash(plan.Path)), document); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	payload := []byte("seed")
	seedID, _ := artifact.IdentifyBytes(artifact.KindEvidence, payload)
	head, err := store.Commit(t.Context(), artifact.Batch{Key: "seed", Artifacts: []artifact.Descriptor{{
		ID: seedID, Size: uint64(len(payload)), MediaType: "application/octet-stream",
	}}})
	if err != nil {
		t.Fatal(err)
	}
	_, sequence := store.Head()
	evidence, err := closurescan.NewCensusEvidence(closurescan.Census{
		Schema: closurescan.CensusSchema, Source: strings.Repeat("a", 64),
	}, head, sequence, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := evidence.Batch(nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := bindCampaignCensus(root, &output); err != nil {
		t.Fatal(err)
	}
	bound, err := plan.Load(filepath.Join(root, filepath.FromSlash(plan.Path)))
	if err != nil || bound.Census == nil || *bound.Census != evidence.ID || !strings.Contains(output.String(), evidence.ID.String()) {
		t.Fatalf("bound census = %+v, output=%q, %v", bound.Census, output.String(), err)
	}
}
