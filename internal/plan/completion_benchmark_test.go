package plan

import (
	"path/filepath"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// BenchmarkCompletionAuthorityRepository measures full replay against the local
// repository and a read-only store. It never fabricates completion evidence.
func BenchmarkCompletionAuthorityRepository(b *testing.B) {
	root := testutil.RepoRoot(b)
	document, err := Load(filepath.Join(root, filepath.FromSlash(Path)))
	if err != nil {
		b.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(filepath.Join(root, "overgodb-store"))
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	_, revision, err := resolveCompletionRevision(b.Context(), root, "HEAD")
	if err != nil {
		b.Fatal(err)
	}
	b.Logf("revision=%s store=%s", revision, filepath.Join(root, "overgodb-store"))
	b.ReportAllocs()
	var authority CompletionAuthority
	for b.Loop() {
		authority, err = ResolveCompletionAuthority(b.Context(), root, revision, document, store)
		if err != nil {
			b.Fatal(err)
		}
	}
	digest, err := completionAuthorityDigest(authority)
	if err != nil {
		b.Fatal(err)
	}
	b.Logf("authority=%s completed=%d retired=%d", digest, len(authority.completedReferences), len(authority.retiredItems))
}
