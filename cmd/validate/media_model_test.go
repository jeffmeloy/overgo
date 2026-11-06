//go:build modeltest

package main

import (
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestMediaMeasurementRetainedProjection projects the frozen protocol's
// model and task cells through the live store's retained verifiers: each
// measured cell's run and gate agree on recipe and commit, and at least one
// cell is measured. It reads the live store, so it runs in the improvement
// cycle; TestMediaMeasurementProjection holds the projection itself on a
// scratch store at the landing.
func TestMediaMeasurementRetainedProjection(t *testing.T) {
	t.Parallel()
	scope := testcheck.Must(resolveMediaReportScope("image-video"))(t)
	root := testutil.RepoRoot(t)
	roots := testcheck.Must(dataroot.Resolve(root))(t)
	retained := overgodbtest.OpenReadOnly(t, retainedReferenceStore(roots.Store))
	var protocol imageVideoProtocol
	testcheck.NoError(t, jsonfile.DecodeStrict(filepath.Join(root, "docs/image_video_protocol.json"), &protocol))
	var entries []discovery.CatalogEntry
	seen := map[string]bool{}
	for _, cell := range protocol.Cases {
		key := cell.Model.String() + "/" + string(cell.Task)
		if seen[key] {
			continue
		}
		seen[key] = true
		entries = append(entries, discovery.CatalogEntry{Model: cell.Model, Capabilities: []discovery.Capability{{Task: cell.Task, Recipe: cell.Recipe}}})
	}
	rows, err := loadMediaReportRows(t.Context(), retained, entries, scope, nil, nil)
	if err != nil || len(rows) != len(seen) || len(rows) == 0 {
		t.Fatal("retained projection denominator differs", err)
	}
	measured := 0
	for _, row := range rows {
		if row.stale != "" {
			continue
		}
		if row.verification.Run.Recipe != row.activation.Definition.ID || row.verification.Gate.Recipe != row.activation.Definition.ID || row.verification.Run.CodeCommit != row.verification.Gate.CodeCommit {
			t.Fatal("retained verifier pair differs")
		}
		measured++
	}
	if measured == 0 {
		t.Fatal("no retained measurement exercised")
	}
	t.Logf("%d retained model/task cells: %d measured, %d stale; no acquisition or current-quality promotion", len(rows), measured, len(rows)-measured)
}
