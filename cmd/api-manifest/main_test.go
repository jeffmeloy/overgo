package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/apimanifest"
	"overgo/internal/artifact"
)

// TestUpdateLeavesACurrentManifestUntouched holds -update to writing only a
// change: the same bytes leave the file and its time alone, since go test
// reruns every suite that reads it when the time moves; new bytes land.
func TestUpdateLeavesACurrentManifestUntouched(t *testing.T) {
	t.Parallel()
	file := filepath.Join(t.TempDir(), "api_manifest.json")
	if written, err := writeChanged(file, []byte("first\n")); err != nil || !written {
		t.Fatalf("an absent manifest: written %t, %v", written, err)
	}
	past := time.Now().Add(-time.Hour)
	if err := os.Chtimes(file, past, past); err != nil {
		t.Fatal(err)
	}
	if written, err := writeChanged(file, []byte("first\n")); err != nil || written {
		t.Fatalf("a current manifest: written %t, %v", written, err)
	}
	if info, err := os.Stat(file); err != nil || !info.ModTime().Equal(past) {
		t.Fatalf("a current manifest's time moved: %v, %v", info.ModTime(), err)
	}
	if written, err := writeChanged(file, []byte("second\n")); err != nil || !written {
		t.Fatalf("a changed manifest: written %t, %v", written, err)
	}
	if data, err := os.ReadFile(file); err != nil || string(data) != "second\n" {
		t.Fatalf("the changed manifest holds %q, %v", data, err)
	}
}

func TestCompileDocumentsPreservesSourceAuthority(t *testing.T) {
	declaration := apimanifest.DocumentDeclaration{
		Name: "fixture", Owner: "overgo/internal/fixture.contract", VersionOwner: "overgo/internal/fixture.version",
		Kind: artifact.KindOutput, MediaType: artifact.JSONMediaType, Schema: "overgo/fixture/v1",
		BuildContexts: []string{"windows-amd64"}, Source: "internal/fixture/document.go", SourceIdentity: "source-identity",
	}
	documents := compileDocuments([]apimanifest.DocumentDeclaration{declaration})
	if len(documents) != 1 || documents[0].SourceIdentity != declaration.SourceIdentity || documents[0].VersionOwner != declaration.VersionOwner {
		t.Fatalf("compiled documents = %+v", documents)
	}
	documents[0].BuildContexts[0] = "changed"
	if declaration.BuildContexts[0] == "changed" {
		t.Fatal("projection aliases source build contexts")
	}
}
