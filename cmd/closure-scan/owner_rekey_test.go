package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
)

// TestClosureOwnerRekeysOnlyVerifiedRebinds keeps the source-owner check
// alive on every commit path: a document naming a version of its file that
// nobody reviewed is refused, where re-keying it on every commit once let it
// through quietly. Only a document the import's rebind matched at a current
// site is re-keyed to the file as it is now, with its reviewed facts intact.
func TestClosureOwnerRekeysOnlyVerifiedRebinds(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	path := filepath.Join(root, "internal", "policy", "policy.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("package policy\nconst Policy = 7\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	candidates, err := closurescan.ScanRoot(root, closurescan.CandidateConstants)
	if err != nil || len(candidates) != 1 {
		t.Fatalf("candidates = (%d, %v)", len(candidates), err)
	}
	binding, err := candidates[0].Binding()
	if err != nil {
		t.Fatal(err)
	}
	document, err := closureledger.New(
		candidates[0].Name, candidates[0].ValueJSON(), closureledger.TierImplementation, closureledger.StatusClosed,
		"Fixture policy remains explicit.", []closureledger.SourceBinding{binding},
		"Replace with fixture authority.", "Fixture contract change.", binding.Owner,
	)
	if err != nil {
		t.Fatal(err)
	}
	// The file changes and the declaration does not move: the document now
	// names a version of the file nobody reviewed.
	if err := os.WriteFile(path, []byte("package policy\nconst Policy = 7\n\n// A later edit elsewhere in the file.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	storePath := filepath.Join(root, "store")
	if _, _, err := commitClosureDocuments(
		root, storePath, closurePublishOperation, []closureledger.Document{document}, nil, nil,
	); err == nil || !strings.Contains(err.Error(), "inconsistent source owner") {
		t.Fatalf("publishing an unreviewed owner = %v, want the owner refused", err)
	}
	rebound, err := currentClosureOwners(root, []closureledger.Document{document})
	if err != nil || len(rebound) != 1 {
		t.Fatalf("re-key = (%d, %v)", len(rebound), err)
	}
	current, _, _, err := fileArtifact(root, binding.File)
	if err != nil {
		t.Fatal(err)
	}
	moved := rebound[0]
	if moved.Bindings[0].Owner != current || moved.Fixture != current || !bytes.Equal(moved.Value, document.Value) ||
		moved.Understanding != document.Understanding || moved.Bindings[0].Line != binding.Line {
		t.Fatalf("re-keyed document = %+v, want owner %v with the reviewed facts unchanged", moved, current)
	}
	if _, _, err := commitClosureDocuments(
		root, storePath, closureRebindOperation, rebound, nil, nil,
	); err != nil {
		t.Fatalf("committing the verified re-key: %v", err)
	}
}
