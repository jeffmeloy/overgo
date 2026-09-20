package overgodb

import (
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

// releaseClaimSchema marks the unreachable claim a release must leave alone.
const releaseClaimSchema = "test/claim/v1"

// releasable admits the retention test schema and nothing else.
func releasable(descriptor artifact.Descriptor) bool { return descriptor.Schema == "test/retention/v1" }

// releaseFixture holds one store with an aliased content, an unreachable
// cache the policy admits and an unreachable claim it does not.
type releaseFixture struct {
	root   string
	store  *Store
	rooted artifact.Content
	orphan artifact.Content
	claim  artifact.Content
}

func newReleaseFixture(t *testing.T) *releaseFixture {
	t.Helper()
	ctx := t.Context()
	root := filepath.Join(t.TempDir(), "store")
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	fixture := &releaseFixture{
		root:   root,
		store:  store,
		rooted: retentionContent(t, artifact.KindEvidence, map[string]any{"kept": "aliased evidence"}),
		orphan: retentionContent(t, artifact.KindOutput, map[string]any{"dropped": "unreachable output"}),
		claim:  retentionContent(t, artifact.KindEvidence, map[string]any{"kept": "unreachable claim"}),
	}
	fixture.claim.Descriptor.Schema = releaseClaimSchema
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:      "release/rooted",
		Contents: []artifact.Content{fixture.rooted},
		Aliases:  []artifact.AliasBinding{{Name: "release/root", Target: fixture.rooted.Descriptor.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:      "release/orphan",
		Contents: []artifact.Content{fixture.orphan, fixture.claim},
	}); err != nil {
		t.Fatal(err)
	}
	return fixture
}

// TestReleaseKeepsChainAuthority releases the orphan's bytes in place: the
// chain gains one commit and keeps every earlier identity, the orphan's
// introduction stays answerable at its original commit, its bytes and blob
// are gone, the aliased content and the unreachable claim are untouched, and
// a rerun releases nothing.
func TestReleaseKeepsChainAuthority(t *testing.T) {
	ctx := t.Context()
	fixture := newReleaseFixture(t)
	store := fixture.store
	orphan := fixture.orphan.Descriptor.ID
	headBefore, sequenceBefore := store.Head()
	introduction, found, err := store.ArtifactIntroduction(ctx, orphan)
	if err != nil || !found {
		t.Fatalf("orphan introduction = (%v, %v)", found, err)
	}
	blob, err := store.blobs.path(orphan)
	if err != nil {
		t.Fatal(err)
	}

	report, err := Release(ctx, store, nil, RetentionPolicy{}, releasable)
	if err != nil {
		t.Fatal(err)
	}
	if report.Unreachable != 2 || report.Released != 1 || report.ReleasedBytes != int64(fixture.orphan.Descriptor.Size) ||
		report.Commits != 1 || report.BlobsRemoved != 1 || report.BlobsAbsent != 0 ||
		report.HeadBefore != headBefore || report.SequenceBefore != sequenceBefore ||
		report.SequenceAfter != sequenceBefore+1 || report.Retained != 1 {
		t.Fatalf("release report = %+v", report)
	}
	if _, statErr := os.Stat(blob); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("released blob still on disk: %v", statErr)
	}
	// Every commit before the release keeps its identity at its sequence.
	for sequence := uint64(1); sequence <= sequenceBefore; sequence++ {
		commit, found, err := store.CommitAt(ctx, sequence)
		if err != nil || !found {
			t.Fatalf("commit %d after release = (%v, %v)", sequence, found, err)
		}
		if sequence == sequenceBefore && commit.ID != headBefore {
			t.Fatalf("pre-release head %s became %s", headBefore, commit.ID)
		}
	}
	assertReleased(t, store, fixture, introduction)

	again, err := Release(ctx, store, nil, RetentionPolicy{}, releasable)
	if err != nil {
		t.Fatal(err)
	}
	if again.Unreachable != 1 || again.Released != 0 || again.Commits != 0 || again.SequenceAfter != report.SequenceAfter {
		t.Fatalf("second release changed the store: %+v", again)
	}
}

// assertReleased holds the released state to its contract from every
// reader: absent bytes, an intact introduction, a typed exact-commit read,
// and the aliased content untouched.
func assertReleased(t *testing.T, store *Store, fixture *releaseFixture, introduction ArtifactIntroduction) {
	t.Helper()
	ctx := t.Context()
	orphan := fixture.orphan.Descriptor.ID
	if has, err := store.HasContent(ctx, orphan); err != nil || has {
		t.Fatalf("released content still present: (%v, %v)", has, err)
	}
	if _, _, found, err := store.OpenContent(ctx, orphan); err != nil || found {
		t.Fatalf("released content opened: (%v, %v)", found, err)
	}
	if _, found, err := store.Artifact(ctx, orphan); err != nil || !found {
		t.Fatalf("released artifact lost its descriptor: (%v, %v)", found, err)
	}
	after, found, err := store.ArtifactIntroduction(ctx, orphan)
	if err != nil || !found || after != introduction {
		t.Fatalf("introduction after release = (%+v, %v, %v), want %+v", after, found, err, introduction)
	}
	if _, _, err := store.CommitDeltaAt(ctx, introduction.Sequence); !errors.Is(err, ErrContentReleased) {
		t.Fatalf("exact read of the introducing commit = %v, want ErrContentReleased", err)
	}
	if err := store.VisitContents(ctx, []artifact.ID{orphan}, func(artifact.Descriptor, io.Reader) error { return nil }); !errors.Is(err, ErrContentReleased) {
		t.Fatalf("visit of released content = %v, want ErrContentReleased", err)
	}
	for name, kept := range map[string]artifact.ID{"aliased": fixture.rooted.Descriptor.ID, "unreachable claim": fixture.claim.Descriptor.ID} {
		_, reader, found, err := store.OpenContent(ctx, kept)
		if err != nil || !found {
			t.Fatalf("%s content after release = (%v, %v)", name, found, err)
		}
		if closer, ok := reader.(io.Closer); ok {
			_ = closer.Close()
		}
	}
}

// TestReleaseSurvivesReplayAndCheckpoints reopens the released store through
// the checkpoint set, the monolithic snapshot and a full journal replay; each
// path reproduces the released state and the intact introduction.
func TestReleaseSurvivesReplayAndCheckpoints(t *testing.T) {
	ctx := t.Context()
	fixture := newReleaseFixture(t)
	introduction, _, err := fixture.store.ArtifactIntroduction(ctx, fixture.orphan.Descriptor.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Release(ctx, fixture.store, nil, RetentionPolicy{}, releasable); err != nil {
		t.Fatal(err)
	}
	head, sequence := fixture.store.Head()
	if err := fixture.store.Close(); err != nil {
		t.Fatal(err)
	}
	reopen := func(name string, prepare func()) {
		t.Helper()
		prepare()
		store, err := OpenReadOnly(fixture.root)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		defer store.Close()
		if replayHead, replaySequence := store.Head(); replayHead != head || replaySequence != sequence {
			t.Fatalf("%s: head (%s, %d), want (%s, %d)", name, replayHead, replaySequence, head, sequence)
		}
		assertReleased(t, store, fixture, introduction)
	}
	reopen("checkpoints", func() {})
	reopen("snapshot", func() {
		if err := os.RemoveAll(filepath.Join(fixture.root, checkpointDirectory)); err != nil {
			t.Fatal(err)
		}
	})
	reopen("journal replay", func() {
		if err := os.RemoveAll(filepath.Join(fixture.root, snapshotDirectory)); err != nil {
			t.Fatal(err)
		}
	})
}

// TestReleaseRefusals binds the two refusals: bytes that were never
// durable, and the current target of an alias.
func TestReleaseRefusals(t *testing.T) {
	ctx := t.Context()
	fixture := newReleaseFixture(t)
	never := retentionContent(t, artifact.KindOutput, map[string]any{"never": "durable"})
	if _, err := fixture.store.Commit(ctx, artifact.Batch{
		Key: "release/never", Releases: []artifact.ID{never.Descriptor.ID},
	}); err == nil || !contains(err.Error(), "never durable") {
		t.Fatalf("release of never-durable content = %v", err)
	}
	if _, err := fixture.store.Commit(ctx, artifact.Batch{
		Key: "release/aliased", Releases: []artifact.ID{fixture.rooted.Descriptor.ID},
	}); err == nil || !contains(err.Error(), "alias target") {
		t.Fatalf("release of an alias target = %v", err)
	}
	if _, err := fixture.store.Commit(ctx, artifact.Batch{
		Key: "release/contradiction", Contents: []artifact.Content{never}, Releases: []artifact.ID{never.Descriptor.ID},
	}); err == nil || !contains(err.Error(), "introduces and releases") {
		t.Fatalf("batch introducing and releasing one content = %v", err)
	}
	if _, sequence := fixture.store.Head(); sequence != 2 {
		t.Fatalf("refused releases advanced the chain to %d", sequence)
	}
}

// TestReleaseThenBackupAndRebuild proves the two whole-store operations read
// the released state correctly: a backup needs no released blob, and a
// rebuild carries the released identity without bytes.
func TestReleaseThenBackupAndRebuild(t *testing.T) {
	ctx := t.Context()
	fixture := newReleaseFixture(t)
	if _, err := Release(ctx, fixture.store, nil, RetentionPolicy{}, releasable); err != nil {
		t.Fatal(err)
	}
	backup := filepath.Join(t.TempDir(), "backup")
	report, err := fixture.store.Backup(ctx, backup)
	if err != nil {
		t.Fatal(err)
	}
	seal, err := ReadBackupSeal(backup)
	if err != nil {
		t.Fatal(err)
	}
	if head, sequence := fixture.store.Head(); seal.Head != head || seal.Sequence != sequence || report.Head != head {
		t.Fatalf("backup seal %+v does not name the released head (%s, %d)", seal, head, sequence)
	}
	rebuilt, err := Rebuild(ctx, fixture.store, filepath.Join(t.TempDir(), "rebuilt"), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	// The aliased content and the unreachable claim carry bytes; the
	// released orphan carries only its identity.
	if rebuilt.ContentsKept != 2 || rebuilt.ContentsDropped != 1 || rebuilt.Artifacts != 3 {
		t.Fatalf("rebuild of a released store = %+v", rebuilt)
	}
}

// TestReleasedContentReintroduces commits the released bytes again: the
// content is durable once more under a fresh introduction.
func TestReleasedContentReintroduces(t *testing.T) {
	ctx := t.Context()
	fixture := newReleaseFixture(t)
	if _, err := Release(ctx, fixture.store, nil, RetentionPolicy{}, releasable); err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.store.Commit(ctx, artifact.Batch{
		Key: "release/reintroduce", Contents: []artifact.Content{fixture.orphan},
	}); err != nil {
		t.Fatal(err)
	}
	_, sequence := fixture.store.Head()
	introduction, found, err := fixture.store.ArtifactIntroduction(ctx, fixture.orphan.Descriptor.ID)
	if err != nil || !found || introduction.Sequence != sequence {
		t.Fatalf("reintroduction = (%+v, %v, %v), want sequence %d", introduction, found, err, sequence)
	}
	if has, err := fixture.store.HasContent(ctx, fixture.orphan.Descriptor.ID); err != nil || !has {
		t.Fatalf("reintroduced content absent: (%v, %v)", has, err)
	}
}

// TestReleaseFollowsChildrenOfPolicyRecords keeps a dependent that only a
// downward walk from its aliased parent reaches: with the parent's schema
// marked for children the dependent is live, without it the dependent is
// an unreachable cache and goes.
func TestReleaseFollowsChildrenOfPolicyRecords(t *testing.T) {
	ctx := t.Context()
	store, err := Open(filepath.Join(t.TempDir(), "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	parent := retentionContent(t, artifact.KindEvidence, map[string]any{"role": "preparation"})
	parent.Descriptor.Schema = "test/root/v1"
	dependent := retentionContent(t, artifact.KindEvidence, map[string]any{"role": "finalization"})
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:      "release/chain",
		Contents: []artifact.Content{parent, dependent},
		Lineage:  []artifact.Lineage{{Child: dependent.Descriptor.ID, Parent: parent.Descriptor.ID, Relation: artifact.RelationDerivedFrom}},
		Aliases:  []artifact.AliasBinding{{Name: "release/chain-root", Target: parent.Descriptor.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	followed := RetentionPolicy{FollowChildren: func(descriptor artifact.Descriptor) bool { return descriptor.Schema == "test/root/v1" }}
	report, err := Release(ctx, store, nil, followed, releasable)
	if err != nil {
		t.Fatal(err)
	}
	if report.Unreachable != 0 || report.Released != 0 || report.Commits != 0 || report.Retained != 2 {
		t.Fatalf("release with followed children = %+v", report)
	}
	report, err = Release(ctx, store, nil, RetentionPolicy{}, releasable)
	if err != nil {
		t.Fatal(err)
	}
	if report.Unreachable != 1 || report.Released != 1 || report.Commits != 1 {
		t.Fatalf("release without followed children = %+v", report)
	}
}

// TestReleaseIsTheOnlyRetentionWriter parses retention.go and holds its
// surface: Release is the only exported operation and the release writer the
// only function that commits, so a rewrite path that starts a new commit
// chain cannot return unnoticed.
func TestReleaseIsTheOnlyRetentionWriter(t *testing.T) {
	file, err := parser.ParseFile(token.NewFileSet(), "retention.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var operations, committers []string
	for _, declaration := range file.Decls {
		function, ok := declaration.(*ast.FuncDecl)
		if !ok {
			continue
		}
		if function.Recv == nil && function.Name.IsExported() {
			operations = append(operations, function.Name.Name)
		}
		ast.Inspect(function, func(node ast.Node) bool {
			call, isCall := node.(*ast.CallExpr)
			if !isCall {
				return true
			}
			if selector, isSelector := call.Fun.(*ast.SelectorExpr); isSelector && selector.Sel.Name == "Commit" {
				committers = append(committers, function.Name.Name)
			}
			return true
		})
	}
	if !slices.Equal(operations, []string{"Release"}) || !slices.Equal(committers, []string{"commit"}) {
		t.Fatalf("retention surface: operations=%v committers=%v", operations, committers)
	}
}

func contains(text, want string) bool { return strings.Contains(text, want) }
