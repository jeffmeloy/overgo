package repoanalysis

import (
	"go/ast"
	"os"
	"path/filepath"
	"testing"
)

func TestSourceSnapshotParseReuse(t *testing.T) {
	root := t.TempDir()
	content := []byte("package p\nfunc same(v int) int { return v + 1 }\n")
	for _, relative := range []string{"a/a.go", "b/b.go"} {
		path := filepath.Join(root, filepath.FromSlash(relative))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, content, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	snapshot, err := DiscoverGo(root, "a", "b")
	if err != nil {
		t.Fatal(err)
	}
	first, _ := snapshot.Files[0].Syntax()
	second, _ := snapshot.Files[1].Syntax()
	if len(snapshot.Files) != 2 || first != second || snapshot.Files[0].ContentID != snapshot.Files[1].ContentID {
		t.Fatal("identical source content was not reused")
	}
}

func TestSourceSnapshotOverlay(t *testing.T) {
	root := t.TempDir()
	for name, content := range map[string]string{
		"a/a.go": "package a\nfunc Before() {}\n",
		"b/b.go": "package b\n",
	} {
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	snapshot, err := DiscoverGo(root, "a", "b")
	if err != nil {
		t.Fatal(err)
	}
	overlaid, err := snapshot.Overlay(map[string][]byte{
		"a/a.go":      []byte("package a\nfunc After() {}\n"),
		"b/b.go":      nil,
		"c/c_test.go": []byte("package c\nfunc TestAdded() {}\n"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Files) != 2 || len(overlaid.Files) != 2 || overlaid.Files[0].Path != "a/a.go" ||
		overlaid.Files[1].Path != "c/c_test.go" || !overlaid.Files[1].Test {
		t.Fatalf("overlay = %+v; original = %+v", overlaid.Files, snapshot.Files)
	}
	file, err := overlaid.Files[0].Syntax()
	if err != nil || file.Decls[0].(*ast.FuncDecl).Name.Name != "After" {
		t.Fatalf("overlaid syntax = %+v, %v", file, err)
	}
	if _, err := snapshot.Overlay(map[string][]byte{"../escape.go": []byte("package bad")}); err == nil {
		t.Fatal("escaping overlay passed")
	}
}
