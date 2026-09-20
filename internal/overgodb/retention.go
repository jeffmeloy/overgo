package overgodb

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

// StoreLocalAliasPrefix marks live authorities whose meaning depends on this
// store's physical commit history. Retention must refuse these aliases unless
// the caller judges them at rest, because a physical operation cannot
// preserve a receipt that is still being written.
const StoreLocalAliasPrefix = "store-local/"

// StoreLocalAdmission decides whether one store-local alias may cross a
// rebuild or a release. The storage layer cannot judge a store-local
// authority's rest state -- the document schemas live above it -- so the
// caller supplies the judgment: return nil to admit the alias, or an error
// naming why the authority is still live. A nil admission refuses every
// store-local alias.
type StoreLocalAdmission func(name string, target artifact.ID) error

// admitStoreLocalAliases applies one admission to every store-local alias
// view, returning the refusal that stops the physical operation.
func admitStoreLocalAliases(operation string, aliases []AliasView, admit StoreLocalAdmission) error {
	for _, alias := range aliases {
		if !strings.HasPrefix(alias.Name, StoreLocalAliasPrefix) {
			continue
		}
		if admit == nil {
			return fmt.Errorf("overgodb: live store-local alias %q prevents %s", alias.Name, operation)
		}
		if err := admit(alias.Name, alias.Target); err != nil {
			return fmt.Errorf("overgodb: store-local alias %q refuses %s: %w", alias.Name, operation, err)
		}
	}
	return nil
}

// RetentionPolicy names what a live set is rooted at beyond the aliases and
// where it grows downward. Roots are identities committed documents and
// commit messages pin without an alias. FollowChildren marks records whose
// dependents consumers discover downward -- a gate preparation's
// finalizations, a gate result's attempts -- so the closure follows their
// lineage children as well as everyone's parents; nil follows no children.
type RetentionPolicy struct {
	Roots          []artifact.ID
	FollowChildren func(artifact.Descriptor) bool
}

// liveSet is one reachability computation over a store: the catalog facts
// copied under the read lock and everything the alias roots and the policy's
// roots reach through manifests, lineage, causality and the followed children.
type liveSet struct {
	descriptors   map[artifact.ID]artifact.Descriptor
	manifests     map[artifact.ID]artifact.Manifest
	contents      map[artifact.ID]bool
	parents       map[artifact.ID][]relationKey
	children      map[artifact.ID][]relationKey
	causality     map[artifact.ID]artifact.CausalLink
	retained      map[artifact.ID]bool
	pinnedRoots   int
	pinnedMissing int
}

// collectLiveSet copies the catalog facts and marks the reachable closure.
// A pinned root the store lacks is counted, never invented.
func collectLiveSet(ctx context.Context, source *Store, operation string, admit StoreLocalAdmission, policy RetentionPolicy) (liveSet, error) {
	live := liveSet{
		descriptors: map[artifact.ID]artifact.Descriptor{},
		manifests:   map[artifact.ID]artifact.Manifest{},
		contents:    map[artifact.ID]bool{},
		parents:     map[artifact.ID][]relationKey{},
		children:    map[artifact.ID][]relationKey{},
		causality:   map[artifact.ID]artifact.CausalLink{},
		retained:    map[artifact.ID]bool{},
	}
	if err := source.Refresh(ctx); err != nil {
		return live, err
	}
	source.mu.RLock()
	if err := source.ready(false); err != nil {
		source.mu.RUnlock()
		return live, err
	}
	aliases := source.state.aliasViews("")
	if err := admitStoreLocalAliases(operation, aliases, admit); err != nil {
		source.mu.RUnlock()
		return live, err
	}
	for execution, link := range source.state.causality.records {
		live.causality[execution] = link.Clone()
	}
	for id, record := range source.state.artifacts.records {
		live.descriptors[id] = record.descriptor
		live.parents[id] = slices.Clone(source.state.lineage.parentsOf(id))
		if policy.FollowChildren != nil && policy.FollowChildren(record.descriptor) {
			live.children[id] = slices.Clone(source.state.lineage.childrenOf(id))
		}
		if source.state.contents.has(id) {
			live.contents[id] = true
		}
		if record.hasManifest {
			live.manifests[id] = record.manifest.Clone()
		}
	}
	source.mu.RUnlock()

	var retain func(artifact.ID)
	retain = func(id artifact.ID) {
		if live.retained[id] {
			return
		}
		if _, known := live.descriptors[id]; !known {
			return
		}
		live.retained[id] = true
		if manifest, ok := live.manifests[id]; ok {
			for _, component := range manifest.Components {
				retain(component.Artifact)
			}
		}
		for _, edge := range live.parents[id] {
			retain(edge.parent)
		}
		for _, edge := range live.children[id] {
			retain(edge.child)
		}
		if link, found := live.causality[id]; found {
			retain(link.Root)
			if link.Subject.Valid() {
				retain(link.Subject)
			}
			for _, evidence := range link.Motivation {
				retain(evidence)
			}
		}
	}
	for _, alias := range aliases {
		retain(alias.Target)
	}
	for _, root := range policy.Roots {
		live.pinnedRoots++
		if _, known := live.descriptors[root]; !known {
			live.pinnedMissing++
			continue
		}
		retain(root)
	}
	return live, nil
}

// ReleaseReport counts one in-place release of content bytes.
type ReleaseReport struct {
	// Retained counts the artifacts the alias and pinned roots reach.
	Retained      int
	PinnedRoots   int
	PinnedMissing int
	// Unreachable counts the durable contents the roots do not reach;
	// Released counts those of them the caller's classes admitted, and
	// ReleasedBytes their descriptor sizes.
	Unreachable   int
	Released      int
	ReleasedBytes int64
	// Commits counts the release commits appended to the chain.
	Commits int
	// Inline counts released contents whose bytes sit inline in a legacy
	// journal frame: the fact is released, the frame stays.
	Inline int
	// BlobsRemoved counts blob files deleted, this run's releases and any
	// earlier release's blobs still present; BlobsAbsent counts released
	// blob contents whose file was already gone.
	BlobsRemoved int
	BlobsAbsent  int
	HeadBefore   artifact.CommitID
	HeadAfter    artifact.CommitID
	// SequenceBefore is the chain length the backup seal must match;
	// SequenceAfter is the head the release commits advanced it to.
	SequenceBefore uint64
	SequenceAfter  uint64
}

// Release releases, in place, the bytes of the durable contents that the
// caller's releasable classes admit and that the policy's live set does not
// reach. Reachability from aliases alone is not liveness in this store:
// consumers also discover records through commit-message trailers, through
// lineage children (a preparation's finalization, a gate result's attempts)
// and through content-addressed memo keys, so the policy supplies those
// roots and the records whose children to follow, and the caller names the
// classes whose bytes are re-derivable; an unreachable record of any other
// class is still a claim and stays. The commit chain stays whole, so every
// commit-coordinate binding outside the store stays exact: the release is
// appended as commits, blob files go only once the journal says their bytes
// are released, and a snapshot then refreshes the checkpoints. A second run
// over the same store releases nothing and appends nothing. A store is never
// rewritten into a fresh journal to compact it: a rewrite is a new commit
// chain, and Rebuild owns that migration.
func Release(ctx context.Context, store *Store, admit StoreLocalAdmission, policy RetentionPolicy, releasable func(artifact.Descriptor) bool) (ReleaseReport, error) {
	report := ReleaseReport{}
	if store == nil || releasable == nil {
		return report, errors.New("overgodb retention: release requires a store and releasable classes")
	}
	live, err := collectLiveSet(ctx, store, "release", admit, policy)
	if err != nil {
		return report, err
	}
	report.Retained, report.PinnedRoots, report.PinnedMissing = len(live.retained), live.pinnedRoots, live.pinnedMissing
	report.HeadBefore, report.SequenceBefore = store.Head()
	report.HeadAfter, report.SequenceAfter = report.HeadBefore, report.SequenceBefore
	var releases []artifact.ID
	for id := range live.contents {
		if live.retained[id] {
			continue
		}
		report.Unreachable++
		if descriptor := live.descriptors[id]; releasable(descriptor) {
			releases = append(releases, id)
			report.ReleasedBytes += int64(descriptor.Size)
		}
	}
	slices.SortFunc(releases, artifact.CompareID)
	report.Released = len(releases)
	if len(releases) == 0 {
		return report, nil
	}
	writer := releaseWriter{ctx: ctx, store: store, head: report.HeadBefore, sequence: report.SequenceBefore}
	if err := writer.commit(releases); err != nil {
		return report, err
	}
	report.Commits = writer.commits
	if err := sweepReleasedBlobs(store, &report); err != nil {
		return report, err
	}
	if _, err := store.Snapshot(ctx); err != nil {
		return report, err
	}
	report.HeadAfter, report.SequenceAfter = store.Head()
	return report, nil
}

// releaseWriter appends release facts onto the store's own chain. The key
// names the chain length released, so reruns never collide.
type releaseWriter struct {
	ctx      context.Context
	store    *Store
	head     artifact.CommitID
	sequence uint64
	commits  int
}

// commit appends the releases as one commit when a frame admits them and
// bisects otherwise, so no chunk size is chosen. The expected head refuses
// an interleaved writer.
func (writer *releaseWriter) commit(releases []artifact.ID) error {
	batch := artifact.Batch{
		Key:          fmt.Sprintf("retention/release/%d/%d", writer.sequence, writer.commits+1),
		ExpectedHead: &writer.head,
		Releases:     releases,
	}
	fits, err := writer.store.transactionFits(batch)
	if err != nil {
		return err
	}
	if fits {
		head, err := writer.store.Commit(writer.ctx, batch)
		if err != nil {
			return err
		}
		writer.head = head
		writer.commits++
		return nil
	}
	if len(releases) == 1 {
		return errors.New("overgodb retention: one release exceeds the frame limit")
	}
	middle := len(releases) / 2
	if err := writer.commit(releases[:middle]); err != nil {
		return err
	}
	return writer.commit(releases[middle:])
}

// sweepReleasedBlobs deletes the blob file of every released content, this
// run's and any earlier run's. Bytes leave the disk only after the chain
// says they are released, so a crash between commit and sweep leaves files
// the next sweep removes, never a committed descriptor whose durable bytes
// are missing. Inline legacy content has no file to remove.
func sweepReleasedBlobs(store *Store, report *ReleaseReport) error {
	var blobs []artifact.ID
	store.mu.RLock()
	for id, locator := range store.state.contents.locators {
		switch {
		case locator.released == 0:
		case locator.blob:
			blobs = append(blobs, id)
		default:
			report.Inline++
		}
	}
	store.mu.RUnlock()
	slices.SortFunc(blobs, artifact.CompareID)
	for _, id := range blobs {
		removed, err := store.blobs.remove(id)
		if err != nil {
			return err
		}
		if removed {
			report.BlobsRemoved++
		} else {
			report.BlobsAbsent++
		}
	}
	return nil
}

// String renders the report.
func (report ReleaseReport) String() string {
	return fmt.Sprintf("retained=%d pinned_roots=%d pinned_missing=%d unreachable=%d released=%d released_bytes=%d commits=%d inline=%d blobs_removed=%d blobs_absent=%d sequence %d -> %d",
		report.Retained, report.PinnedRoots, report.PinnedMissing, report.Unreachable, report.Released, report.ReleasedBytes,
		report.Commits, report.Inline, report.BlobsRemoved, report.BlobsAbsent, report.SequenceBefore, report.SequenceAfter)
}
