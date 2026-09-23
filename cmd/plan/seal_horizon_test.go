package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/plan"
)

// TestSealHorizonProvesBeforeItWrites holds the seal to its dry run. A
// receipt is written only after the authority resolved at the commit that
// would carry it, and that commit is unreferenced: HEAD, the index and the
// tracked working tree are where they were. A receipt the authority cannot
// accept -- one that miscounts this history -- fails the dry run, which
// reads it from the unreferenced commit's tree, and leaves nothing written.
func TestSealHorizonProvesBeforeItWrites(t *testing.T) {
	t.Parallel()
	document := plan.Plan{Items: []plan.Item{{
		ID: "row", Status: plan.StatusOpen, Steps: []plan.Step{{ID: "do", Status: plan.StatusOpen, Verify: "go test ./..."}},
	}}}
	root := initializePlanTestRepository(t, document)
	state := func() string {
		t.Helper()
		head, err := gitOutput(root, "rev-parse", "HEAD")
		if err != nil {
			t.Fatal(err)
		}
		status, err := gitOutput(root, "status", "--porcelain", "--untracked-files=no")
		if err != nil {
			t.Fatal(err)
		}
		return string(head) + string(status)
	}
	before := state()
	receiptPath := filepath.Join(root, filepath.FromSlash(plan.ProofHorizonPath))

	miscounted, err := json.Marshal(plan.ProofHorizon{Commit: strings.TrimSpace(before), Completions: 5})
	if err != nil {
		t.Fatal(err)
	}
	if err := proveProofHorizon(root, document, miscounted); err == nil || !strings.Contains(err.Error(), "sealed 5") {
		t.Fatalf("a receipt that miscounts history survived its dry run: %v", err)
	}
	if _, err := os.Stat(receiptPath); err == nil || state() != before {
		t.Fatalf("a failed dry run left a receipt or moved the repository: %q", state())
	}

	var output bytes.Buffer
	if err := sealProofHorizon(root, &output); err != nil {
		t.Fatal(err)
	}
	var sealed plan.ProofHorizon
	data, err := os.ReadFile(receiptPath)
	if err != nil || json.Unmarshal(data, &sealed) != nil {
		t.Fatalf("sealed receipt = %s, %v", data, err)
	}
	if sealed.Commit != strings.TrimSpace(before) || sealed.Completions != 0 || !strings.Contains(output.String(), sealed.Commit) {
		t.Fatalf("sealed %+v with output %q at %q", sealed, output.String(), before)
	}
	if state() != before {
		t.Fatalf("sealing moved HEAD, the index or a tracked file: %q", state())
	}
}
