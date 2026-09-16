package main

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/codemanifest"
)

func TestCompileDocumentsPreservesSourceAuthority(t *testing.T) {
	declaration := codemanifest.DocumentDeclaration{
		Name: "fixture", Owner: "overgo/internal/fixture.contract", VersionOwner: "overgo/internal/fixture.version",
		Kind: artifact.KindOutput, MediaType: artifact.JSONMediaType, Schema: "overgo/fixture/v1",
		BuildContexts: []string{"windows-amd64"}, Source: "internal/fixture/document.go", SourceIdentity: "source-identity",
	}
	documents := compileDocuments([]codemanifest.DocumentDeclaration{declaration})
	if len(documents) != 1 || documents[0].SourceIdentity != declaration.SourceIdentity || documents[0].VersionOwner != declaration.VersionOwner {
		t.Fatalf("compiled documents = %+v", documents)
	}
	documents[0].BuildContexts[0] = "changed"
	if declaration.BuildContexts[0] == "changed" {
		t.Fatal("projection aliases source build contexts")
	}
}
