package overgodb

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

func TestRebuildRefusesLiveStoreLocalAlias(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	source, err := Open(filepath.Join(root, "source"))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	contract := artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: "application/json", Schema: "test/rebuild-store-local/v1",
	}
	content, err := contract.ContentBytes([]byte(`{"receipt":"physical"}`))
	if err != nil {
		t.Fatal(err)
	}
	localAlias := StoreLocalAliasPrefix + "fixture/current"
	if _, err := source.Commit(ctx, artifact.Batch{
		Key: "test/rebuild/store-local", Contents: []artifact.Content{content},
		Aliases: []artifact.AliasBinding{{Name: localAlias, Target: content.Descriptor.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	destination := filepath.Join(root, "rebuilt")
	if _, err := Rebuild(ctx, source, destination, nil); err == nil || !strings.Contains(err.Error(), localAlias) {
		t.Fatalf("store-local rebuild error = %v", err)
	}
	if _, err := os.Stat(destination); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("refused rebuild created destination: %v", err)
	}
}
