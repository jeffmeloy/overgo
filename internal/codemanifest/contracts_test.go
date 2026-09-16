package codemanifest

import (
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gosource"
	"overgo/internal/repoanalysis"
)

func TestDocumentDeclarationsResolvePackageConstantsAndContexts(t *testing.T) {
	root := t.TempDir()
	files := map[string]string{
		"internal/artifact/artifact.go": `package artifact
type Kind uint8
const (
	KindInvalid Kind = iota
	KindModel
	KindTensorSet
	KindTokenizer
	KindProjector
	KindAdapter
	KindDataset
	KindDatasetShard
	KindCheckpoint
	KindRecipe
	KindOutput
)
const JSONMediaType = "application/json"
type DocumentContract struct { Kind Kind; MediaType, Schema string }
type DocumentCodec[T any] struct { Contract DocumentContract }
func JSONDocumentCodec[T any](string, Kind, string, string, ...any) DocumentCodec[T] { return DocumentCodec[T]{} }
`,
		"internal/example/contracts.go": `package example
import "overgo/internal/artifact"
const outputSchema = "overgo/example/" + "v1"
var outputContract = artifact.DocumentContract{
	Kind: artifact.KindOutput, MediaType: artifact.JSONMediaType, Schema: outputSchema,
}
var resultCodec = artifact.JSONDocumentCodec[struct{}](
	"result", artifact.KindOutput, artifact.JSONMediaType, "overgo/result/v1",
)
var directCodec = artifact.DocumentCodec[struct{}]{Contract: outputContract}
`,
	}
	var paths []string
	for name, content := range files {
		writeGeneratorFixture(t, root, name, content)
		paths = append(paths, name)
	}
	snapshot, err := repoanalysis.LoadGo(root, paths)
	if err != nil {
		t.Fatal(err)
	}
	selection := gosource.BuildSelection{
		Context: "linux/amd64", Root: root, Files: map[string]bool{}, Packages: map[string]string{},
	}
	for _, name := range paths {
		selection.Files[name] = true
		selection.Packages[name] = "overgo/" + name[:len(name)-len("/artifact.go")]
	}
	selection.Packages["internal/example/contracts.go"] = "overgo/internal/example"
	declarations, err := DocumentDeclarations(snapshot, []gosource.BuildSelection{selection})
	if err != nil {
		t.Fatal(err)
	}
	if len(declarations) != 2 {
		t.Fatalf("document declarations = %+v", declarations)
	}
	first := declarations[0]
	if first.Kind != artifact.KindOutput || first.MediaType != artifact.JSONMediaType || first.Schema != outputSchemaFixture || len(first.BuildContexts) != 1 {
		t.Fatalf("first declaration = %+v", first)
	}
}

const outputSchemaFixture = "overgo/example/v1"

func TestRepositoryDocumentDeclarationsResolveKnownAuthorities(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	selection, err := gosource.HostBuildSelection(root, "./internal/...", "./cmd/...")
	if err != nil {
		t.Fatal(err)
	}
	declarations, err := DocumentDeclarations(snapshot, []gosource.BuildSelection{selection})
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]bool{
		"overgo/api-manifest/v1":  false,
		"overgo/code-manifest/v1": false,
	}
	seen := map[string]string{}
	for _, declaration := range declarations {
		key := declaration.Kind.String() + "\x00" + declaration.MediaType + "\x00" + declaration.Schema
		if owner := seen[key]; owner != "" {
			t.Errorf("document contract %q duplicated by %s and %s", declaration.Schema, owner, declaration.Owner)
		}
		seen[key] = declaration.Owner
		if _, known := want[declaration.Schema]; known {
			want[declaration.Schema] = true
		}
	}
	for schema, found := range want {
		if !found {
			t.Errorf("document contract %q absent", schema)
		}
	}
}

func TestDocumentDeclarationsRejectDynamicCodecContracts(t *testing.T) {
	root := t.TempDir()
	name := "internal/example/contracts.go"
	writeGeneratorFixture(t, root, name, `package example
import "overgo/internal/artifact"
func schema() string { return "overgo/example/v1" }
var codec = artifact.JSONDocumentCodec[struct{}]("result", artifact.KindOutput, artifact.JSONMediaType, schema())
`)
	snapshot, err := repoanalysis.LoadGo(root, []string{name})
	if err != nil {
		t.Fatal(err)
	}
	selection := gosource.BuildSelection{
		Context: "linux/amd64", Files: map[string]bool{name: true},
		Packages: map[string]string{name: "overgo/internal/example"},
	}
	if _, err := DocumentDeclarations(snapshot, []gosource.BuildSelection{selection}); err == nil {
		t.Fatal("dynamic codec contract accepted")
	}
}
