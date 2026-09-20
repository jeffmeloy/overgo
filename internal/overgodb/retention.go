package overgodb

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

// StoreLocalAliasPrefix marks live authorities whose meaning depends on this
// store's physical commit history. Compaction must refuse these aliases because
// rebuilding logical facts cannot preserve their original commit receipts.
const StoreLocalAliasPrefix = "store-local/"

// StoreLocalAdmission decides whether one store-local alias may cross a
// rebuild or compaction. The storage layer cannot judge a store-local
// authority's rest state -- the document schemas live above it -- so the
// caller supplies the judgment: return nil to admit the alias into the
// destination, or an error naming why the authority is still live. A nil
// admission keeps the historical behavior of refusing every store-local
// alias.
type StoreLocalAdmission func(name string, target artifact.ID) error

// admitStoreLocalAliases applies one admission to every store-local alias
// view, returning the refusal that stops the physical rewrite.
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

// RetentionReport counts one compaction.
type RetentionReport struct {
	RetainedArtifacts int
	RetainedManifests int
	RetainedContents  int
	Aliases           int
	Lineage           int
	Causality         int
	DroppedArtifacts  int
	// ReclaimableBlobs and ReclaimableBlobBytes report source blob
	// files no retained content references; retention rebuilds a
	// destination and reports reclaimable material, it never deletes.
	ReclaimableBlobs     int
	ReclaimableBlobBytes int64
	// ObsoleteCheckpoints counts source checkpoint files that no
	// longer form a valid anchored set.
	ObsoleteCheckpoints int
	// PinnedRoots counts identities the caller named as roots beyond the
	// aliases (committed documents that pin store evidence); PinnedMissing
	// counts pinned identities the source store does not hold.
	PinnedRoots   int
	PinnedMissing int
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
	locations     map[artifact.ID][]artifact.Location
	causality     map[artifact.ID]artifact.CausalLink
	aliases       []AliasView
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
		locations:   map[artifact.ID][]artifact.Location{},
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
	live.aliases = source.state.aliasViews("")
	if err := admitStoreLocalAliases(operation, live.aliases, admit); err != nil {
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
		live.locations[id] = slices.Clone(source.state.locations.of(id))
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
	for _, alias := range live.aliases {
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

// Compact writes alias-rooted parent closure to a new store.
//
// A rewrite starts a new commit chain: every commit-coordinate binding
// outside the store -- gate trailers, merge receipts, census sequences --
// names commits the destination does not have. Compact is a lineage
// migration, not a compaction of a store in service; ReleaseUnreachable is.
func Compact(ctx context.Context, source *Store, destinationRoot string, admit StoreLocalAdmission) (RetentionReport, error) {
	return CompactWithRoots(ctx, source, destinationRoot, admit, nil)
}

// CompactWithRoots is Compact with extra roots: identities the repository's
// committed documents pin without an alias survive retention when named here,
// and a pinned identity the source lacks is counted, never invented.
func CompactWithRoots(ctx context.Context, source *Store, destinationRoot string, admit StoreLocalAdmission, roots []artifact.ID) (RetentionReport, error) {
	report := RetentionReport{}
	if source == nil {
		return report, errors.New("overgodb retention: nil source store")
	}
	live, err := collectLiveSet(ctx, source, "compaction", admit, RetentionPolicy{Roots: roots})
	if err != nil {
		return report, err
	}
	report.PinnedRoots, report.PinnedMissing = live.pinnedRoots, live.pinnedMissing
	descriptors, manifests, contents := live.descriptors, live.manifests, live.contents
	parents, locations, causality := live.parents, live.locations, live.causality
	aliases, retained := live.aliases, live.retained
	ids := slices.SortedFunc(maps.Keys(retained), artifact.CompareID)
	switch {
	case ids == nil:
		ids = []artifact.ID{}
	}

	destination, err := Open(destinationRoot)
	if err != nil {
		return report, err
	}
	defer destination.Close()
	if _, sequence := destination.Head(); sequence != 0 {
		return report, fmt.Errorf("overgodb retention: destination %s already holds %d commit(s)", destinationRoot, sequence)
	}
	writer := compactionWriter{ctx: ctx, destination: destination}
	contentIDs := make([]artifact.ID, 0, len(contents))
	for _, id := range ids {
		if _, manifest := manifests[id]; manifest {
			continue
		}
		if contents[id] {
			contentIDs = append(contentIDs, id)
			continue
		}
		item := artifact.Batch{Artifacts: []artifact.Descriptor{descriptors[id]}}
		for _, location := range locations[id] {
			item.Locations = append(item.Locations, artifact.LocationEvent{Location: location, Action: artifact.LocationAdd})
		}
		if err := writer.add(item); err != nil {
			return report, err
		}
		report.RetainedArtifacts++
	}
	if err := source.VisitContents(ctx, contentIDs, func(descriptor artifact.Descriptor, reader io.Reader) error {
		data, err := io.ReadAll(reader)
		if err != nil {
			return err
		}
		item := artifact.Batch{Contents: []artifact.Content{{Descriptor: descriptor, Data: data}}}
		for _, location := range locations[descriptor.ID] {
			item.Locations = append(item.Locations, artifact.LocationEvent{Location: location, Action: artifact.LocationAdd})
		}
		if err := writer.add(item); err != nil {
			return err
		}
		report.RetainedArtifacts++
		report.RetainedContents++
		return nil
	}); err != nil {
		return report, err
	}
	if err := writer.flush(); err != nil {
		return report, err
	}
	for _, id := range ids {
		manifest, ok := manifests[id]
		if !ok {
			continue
		}
		item := artifact.Batch{Manifests: []artifact.Manifest{manifest}}
		for _, location := range locations[id] {
			item.Locations = append(item.Locations, artifact.LocationEvent{Location: location, Action: artifact.LocationAdd})
		}
		if err := writer.add(item); err != nil {
			return report, err
		}
		report.RetainedManifests++
	}
	if err := writer.flush(); err != nil {
		return report, err
	}
	manifestLineage := map[relationKey]struct{}{}
	for _, manifest := range manifests {
		for _, edge := range manifest.Lineage() {
			manifestLineage[relationKey{child: edge.Child, parent: edge.Parent, relation: edge.Relation}] = struct{}{}
		}
	}
	for _, id := range ids {
		for _, edge := range parents[id] {
			if !retained[edge.parent] {
				continue
			}
			if _, materialized := manifestLineage[edge]; materialized {
				report.Lineage++
				continue
			}
			if err := writer.add(artifact.Batch{Lineage: []artifact.Lineage{{
				Child: edge.child, Parent: edge.parent, Relation: edge.relation,
			}}}); err != nil {
				return report, err
			}
			report.Lineage++
		}
	}
	if err := writer.flush(); err != nil {
		return report, err
	}
	for _, id := range ids {
		if link, found := causality[id]; found {
			if err := writer.add(artifact.Batch{Causality: []artifact.CausalLink{link}}); err != nil {
				return report, err
			}
			report.Causality++
		}
	}
	if err := writer.flush(); err != nil {
		return report, err
	}
	for _, alias := range aliases {
		if !retained[alias.Target] {
			continue
		}
		if err := writer.add(artifact.Batch{Aliases: []artifact.AliasBinding{{
			Name: alias.Name, Target: alias.Target,
		}}}); err != nil {
			return report, err
		}
		report.Aliases++
	}
	if err := writer.flush(); err != nil {
		return report, err
	}
	if _, err := destination.Snapshot(ctx); err != nil {
		return report, err
	}
	report.DroppedArtifacts = len(descriptors) - report.RetainedArtifacts - report.RetainedManifests
	reportReclaimable(source, retained, contents, &report)
	return report, nil
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
// over the same store releases nothing and appends nothing.
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
	// One item per release lets the writer bisect to what a frame admits,
	// so no chunk size is chosen; the key names the chain length released.
	items := make([]artifact.Batch, len(releases))
	for index, id := range releases {
		items[index] = artifact.Batch{Releases: []artifact.ID{id}}
	}
	writer := compactionWriter{
		ctx: ctx, destination: store, head: report.HeadBefore,
		keyPrefix: fmt.Sprintf("retention/release/%d", report.SequenceBefore),
	}
	if err := writer.commit(items); err != nil {
		return report, err
	}
	report.Commits = writer.chunk
	if err := sweepReleasedBlobs(store, &report); err != nil {
		return report, err
	}
	if _, err := store.Snapshot(ctx); err != nil {
		return report, err
	}
	report.HeadAfter, report.SequenceAfter = store.Head()
	return report, nil
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

// reportReclaimable names the source material canonical reachability
// no longer requires: blob files no retained content references and
// checkpoint files that no longer form a valid anchored set. Retention
// reports; it never deletes in place.
func reportReclaimable(source *Store, retained map[artifact.ID]bool, contents map[artifact.ID]bool, report *RetentionReport) {
	referenced := map[string]bool{}
	for id, hasContent := range contents {
		if hasContent && retained[id] {
			referenced[id.DigestHex()] = true
		}
	}
	root := filepath.Join(source.root, blobDirectory)
	_ = filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return nil
		}
		if !referenced[filepath.Base(path)] {
			report.ReclaimableBlobs++
			if info, infoErr := entry.Info(); infoErr == nil {
				report.ReclaimableBlobBytes += info.Size()
			}
		}
		return nil
	})
	if _, _, valid, _ := loadProjectionCheckpoints(source.root); !valid {
		entries, err := os.ReadDir(filepath.Join(source.root, checkpointDirectory))
		if err == nil {
			report.ObsoleteCheckpoints = len(entries)
		}
	}
}

// compactionWriter commits retention facts to a store in frame-sized
// chunks: compaction's rebuilt facts into a fresh store, or release facts
// onto the store's own chain. keyPrefix names the operation; empty means
// compaction.
type compactionWriter struct {
	ctx         context.Context
	destination *Store
	items       []artifact.Batch
	content     int
	chunk       int
	head        artifact.CommitID
	keyPrefix   string
}

func (writer *compactionWriter) add(item artifact.Batch) error {
	content := 0
	for _, value := range item.Contents {
		content += len(value.Data)
	}
	if content > maxFramePayload {
		return errors.New("overgodb retention: one compacted fact exceeds the frame limit")
	}
	if writer.content > maxFramePayload-content {
		if err := writer.flush(); err != nil {
			return err
		}
	}
	writer.items = append(writer.items, item)
	writer.content += content
	return nil
}

func (writer *compactionWriter) flush() error {
	if len(writer.items) == 0 {
		return nil
	}
	if err := writer.commit(writer.items); err != nil {
		return err
	}
	clear(writer.items)
	writer.items = writer.items[:0]
	writer.content = len(writer.items)
	return nil
}

func (writer *compactionWriter) commit(items []artifact.Batch) error {
	batch := mergeCompactionItems(items)
	batch.Key = writer.nextKey()
	batch.ExpectedHead = &writer.head
	fits, err := writer.destination.transactionFits(batch)
	if err != nil {
		return err
	}
	if fits {
		head, err := writer.destination.Commit(writer.ctx, batch)
		if err != nil {
			return err
		}
		writer.head = head
		writer.chunk++
		return nil
	}
	if len(items) == 1 {
		return errors.New("overgodb retention: one compacted fact exceeds the frame limit")
	}
	middle := len(items) / 2
	if err := writer.commit(items[:middle]); err != nil {
		return err
	}
	return writer.commit(items[middle:])
}

func (writer *compactionWriter) nextKey() string {
	return fmt.Sprintf("%s/%d", cmp.Or(writer.keyPrefix, "retention/compact"), writer.chunk+1)
}

func mergeCompactionItems(items []artifact.Batch) artifact.Batch {
	var artifacts, contents, manifests, lineage, causality, aliases, locations, releases int
	for _, item := range items {
		artifacts += len(item.Artifacts)
		contents += len(item.Contents)
		manifests += len(item.Manifests)
		lineage += len(item.Lineage)
		causality += len(item.Causality)
		aliases += len(item.Aliases)
		locations += len(item.Locations)
		releases += len(item.Releases)
	}
	merged := artifact.Batch{
		Artifacts: make([]artifact.Descriptor, artifacts),
		Contents:  make([]artifact.Content, contents),
		Manifests: make([]artifact.Manifest, manifests),
		Lineage:   make([]artifact.Lineage, lineage),
		Causality: make([]artifact.CausalLink, causality),
		Aliases:   make([]artifact.AliasBinding, aliases),
		Locations: make([]artifact.LocationEvent, locations),
		Releases:  make([]artifact.ID, releases),
	}
	var artifactAt, contentAt, manifestAt, lineageAt, causalityAt, aliasAt, locationAt, releaseAt int
	for _, item := range items {
		artifactAt += copy(merged.Artifacts[artifactAt:], item.Artifacts)
		contentAt += copy(merged.Contents[contentAt:], item.Contents)
		manifestAt += copy(merged.Manifests[manifestAt:], item.Manifests)
		lineageAt += copy(merged.Lineage[lineageAt:], item.Lineage)
		causalityAt += copy(merged.Causality[causalityAt:], item.Causality)
		aliasAt += copy(merged.Aliases[aliasAt:], item.Aliases)
		locationAt += copy(merged.Locations[locationAt:], item.Locations)
		releaseAt += copy(merged.Releases[releaseAt:], item.Releases)
	}
	return merged
}

// String renders the report.
func (report RetentionReport) String() string {
	return fmt.Sprintf("retained artifacts=%d manifests=%d contents=%d aliases=%d lineage=%d causality=%d; dropped=%d reclaimable_blobs=%d reclaimable_blob_bytes=%d obsolete_checkpoints=%d",
		report.RetainedArtifacts, report.RetainedManifests, report.RetainedContents,
		report.Aliases, report.Lineage, report.Causality, report.DroppedArtifacts,
		report.ReclaimableBlobs, report.ReclaimableBlobBytes, report.ObsoleteCheckpoints)
}
