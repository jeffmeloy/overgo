package loop_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
)

// TestRSIRuntimeIsGoOnly proves the loop's strategies and proposals stay
// typed data over a Go-only runtime: the shared Go-only rule holds over the
// live tree, and dynamic library loading outside the enumerated OS-ABI owners
// — a capability grant no strategy document may cause — is refused by the
// same rule the gate runs on every commit.
func TestRSIRuntimeIsGoOnly(t *testing.T) {
	rule, found := closurescan.EntryAuthorityRuleFor(closurescan.EntryAuthorityGoOnly)
	if !found || len(rule.Exceptions) == 0 {
		t.Fatalf("go-only entry authority rule = (found=%t, exceptions=%d)", found, len(rule.Exceptions))
	}
	snapshot, err := repoanalysis.DiscoverGo(filepath.Join("..", ".."), "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := closurescan.ValidateEntryAuthorities(snapshot, []closurescan.EntryAuthorityRule{rule}); err != nil {
		t.Fatalf("live tree violates the go-only runtime authority: %v", err)
	}

	root := t.TempDir()
	rogue := filepath.Join(root, "internal", "rogue")
	if err := os.MkdirAll(rogue, 0o755); err != nil {
		t.Fatal(err)
	}
	source := "package rogue\n\nimport \"syscall\"\n\nfunc bypass() *syscall.LazyDLL {\n\treturn syscall.NewLazyDLL(\"rogue.dll\")\n}\n"
	if err := os.WriteFile(filepath.Join(rogue, "rogue.go"), []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	violation, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := closurescan.ValidateEntryAuthorities(violation, []closurescan.EntryAuthorityRule{rule}); err == nil ||
		!strings.Contains(err.Error(), "internal/rogue/rogue.go bypasses the owner") {
		t.Fatalf("dynamic library loading was not refused: %v", err)
	}

	interpreter := filepath.Join(root, "internal", "rogue", "interpreter")
	if err := os.MkdirAll(interpreter, 0o755); err != nil {
		t.Fatal(err)
	}
	embedded := "package interpreter\n\nimport _ \"github.com/dop251/goja\"\n"
	if err := os.WriteFile(filepath.Join(interpreter, "embedded.go"), []byte(embedded), 0o644); err != nil {
		t.Fatal(err)
	}
	embeddedSnapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := closurescan.ValidateEntryAuthorities(embeddedSnapshot, []closurescan.EntryAuthorityRule{rule}); err == nil ||
		!strings.Contains(err.Error(), "bypasses the owner") {
		t.Fatalf("embedded interpreter import was not refused: %v", err)
	}
}
