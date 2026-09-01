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
	if _, err := Rebuild(ctx, source, destination, nil, nil); err == nil || !strings.Contains(err.Error(), localAlias) {
		t.Fatalf("store-local rebuild error = %v", err)
	}
	if _, err := os.Stat(destination); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("refused rebuild created destination: %v", err)
	}

	// An admission that still names the authority live refuses with its
	// reason, and no destination appears.
	refused := errors.New("authority is mid-flight")
	if _, err := Rebuild(ctx, source, destination, nil, func(name string, target artifact.ID) error {
		if name != localAlias || target != content.Descriptor.ID {
			t.Fatalf("admission saw %s -> %s", name, target)
		}
		return refused
	}); err == nil || !errors.Is(err, refused) {
		t.Fatalf("refusing admission error = %v", err)
	}
	if _, err := os.Stat(destination); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("refused admission created destination: %v", err)
	}

	// An at-rest admission lets the rebuild proceed and carries the
	// store-local alias into the destination bound to the same
	// content-addressed target.
	if _, err := Rebuild(ctx, source, destination, nil, func(string, artifact.ID) error { return nil }); err != nil {
		t.Fatal(err)
	}
	rebuilt, err := Open(destination)
	if err != nil {
		t.Fatal(err)
	}
	defer rebuilt.Close()
	target, found, err := artifact.ResolveAlias(ctx, rebuilt, localAlias)
	if err != nil || !found || target != content.Descriptor.ID {
		t.Fatalf("rebuilt store-local alias = (%s, %v, %v)", target, found, err)
	}

	compacted := filepath.Join(root, "compacted")
	if _, err := Compact(ctx, source, compacted, func(string, artifact.ID) error { return nil }); err != nil {
		t.Fatal(err)
	}
}
