package overgodb

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
)

// TestSmallContentIsInline holds the write path to the size rule: content one
// byte under the limit rides in its frame and leaves no blob file, content at
// the limit is a blob, and both read back identically straight away, after the
// journal seals, and from a store reopened with no snapshot or checkpoint,
// where replay alone places them. An inline content is released like any
// other: its bytes stay in the frame and its readers see it gone.
func TestSmallContentIsInline(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	root := filepath.Join(t.TempDir(), "store")
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	sized := func(size int, fill byte) artifact.Content {
		t.Helper()
		data := bytes.Repeat([]byte{fill}, size)
		id, err := artifact.IdentifyBytes(artifact.KindOutput, data)
		if err != nil {
			t.Fatal(err)
		}
		return artifact.Content{Descriptor: artifact.Descriptor{ID: id, Size: uint64(size), MediaType: "text/plain"}, Data: data}
	}
	small, large := sized(inlineContentLimit-1, 's'), sized(inlineContentLimit, 'l')
	if _, err := store.Commit(ctx, artifact.Batch{Key: "inline/sizes", Contents: []artifact.Content{small, large}}); err != nil {
		t.Fatal(err)
	}
	for content, wantBlob := range map[*artifact.Content]bool{&small: false, &large: true} {
		path, err := store.blobs.path(content.Descriptor.ID)
		if err != nil {
			t.Fatal(err)
		}
		if _, statErr := os.Stat(path); (statErr == nil) != wantBlob {
			t.Fatalf("content of %d bytes: blob file present=%v, want %v", content.Descriptor.Size, statErr == nil, wantBlob)
		}
	}
	read := func(from *Store, when string) {
		t.Helper()
		for _, want := range []artifact.Content{small, large} {
			got, found, err := artifact.ReadContent(ctx, from, want.Descriptor.ID)
			if err != nil || !found || !bytes.Equal(got.Data, want.Data) {
				t.Fatalf("%s: content of %d bytes found=%v equal=%v: %v", when, want.Descriptor.Size, found, bytes.Equal(got.Data, want.Data), err)
			}
		}
	}
	read(store, "after the commit")
	if _, err := store.Snapshot(ctx); err != nil {
		t.Fatal(err)
	}
	read(store, "after the journal sealed")
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	for _, derived := range []string{snapshotDirectory, checkpointDirectory} {
		if err := os.RemoveAll(filepath.Join(root, derived)); err != nil {
			t.Fatal(err)
		}
	}
	store, err = Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	read(store, "after a reopen by full replay")

	report, err := Release(ctx, store, func(string, artifact.ID) error { return nil }, RetentionPolicy{},
		func(descriptor artifact.Descriptor) bool { return descriptor.ID == small.Descriptor.ID })
	if err != nil || report.Released != 1 || report.Inline != 1 {
		t.Fatalf("release of the inline content = %+v, %v", report, err)
	}
	if held, err := store.HasContent(ctx, small.Descriptor.ID); err != nil || held {
		t.Fatalf("a released inline content is still held: %v, %v", held, err)
	}
}
