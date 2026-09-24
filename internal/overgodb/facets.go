package overgodb

import (
	"cmp"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

// Catalog state is composed of seven concrete facet owners. Each facet
// owns one class of catalog fact together with its indexes, conflict
// validation, batch delta, and application; catalogState remains the
// single aggregate transaction view that drives them in commit order.
// No facet is an interface: checkpointing supplies the second consumer
// that would justify one, and it has not landed yet.

// projection is the compiled contract every catalog facet satisfies:
// an explicit name, an explicit schema version, and commit-ordered
// application. Registration is closed-world at compile time --
// projections(state) is the complete enumeration -- with no
// reflection, discovery, or independent write authority. Replay
// drives the contract now; per-projection checkpoints are its second
// consumer.
type projection interface {
	// accept refuses a commit the projection could not apply exactly;
	// it runs before anything becomes durable, so a refused commit
	// publishes nothing anywhere.
	accept(delta artifact.Batch, locators map[artifact.ID]contentLocator, sequence uint64) error
	applyCommit(delta artifact.Batch, locators map[artifact.ID]contentLocator, sequence uint64)
	// checkpoint serializes the projection's complete state canonically:
	// serializing a restored checkpoint reproduces it byte for byte.
	checkpoint() ([]byte, error)
	// restore replaces the projection's state from one checkpoint.
	restore(data []byte) error
}

// registeredProjection binds one facet to its explicit name and
// checkpoint schema version; the registry table is the single
// declaration site for both.
type registeredProjection struct {
	name    string
	version uint16
	view    projection
}

// initialProjectionVersion is every projection's first schema
// version; a projection that changes its checkpoint shape advances
// its own version constant away from this shared origin.
const initialProjectionVersion uint16 = 1

// contentProjectionVersion advances when the content projection gains a new
// persisted fact. Older checkpoints remain acceleration hints and fall back to
// snapshot or journal replay. Version 3 added the release sequence.
const contentProjectionVersion = initialProjectionVersion + 2

// commitProjectionVersion advances when exact journal coordinates become a
// durable part of the commit projection.
const commitProjectionVersion = initialProjectionVersion + 1

// projections enumerates the seven facet projections in commit-
// application order; this table is the closed world and the single
// owner of projection names and versions.
func projections(state *catalogState) []registeredProjection {
	return []registeredProjection{
		{name: "artifacts", version: initialProjectionVersion, view: &state.artifacts},
		{name: "contents", version: contentProjectionVersion, view: &state.contents},
		{name: "lineage", version: initialProjectionVersion, view: &state.lineage},
		{name: "causality", version: CausalityProjectionVersion, view: &state.causality},
		{name: "locations", version: initialProjectionVersion, view: &state.locations},
		{name: "aliases", version: initialProjectionVersion, view: &state.aliases},
		{name: "commits", version: commitProjectionVersion, view: &state.commits},
	}
}

// artifactRecord carries the artifact facet's facts for one identity.
type artifactRecord struct {
	descriptor  artifact.Descriptor
	manifest    artifact.Manifest
	sequence    uint64
	hasManifest bool
	// position is the record's place in bySequence.
	position int
}

// artifactFacet owns descriptors, manifests, and the media, schema,
// and sequence indexes derived from them.
type artifactFacet struct {
	records    map[artifact.ID]*artifactRecord
	byMedia    map[string][]artifact.ID
	bySchema   map[string][]artifact.ID
	byKind     map[artifact.Kind][]artifact.ID
	bySequence []artifact.ID
	// schemaPositions holds each schema's bySequence positions ascending,
	// so a document read walks its schema in commit order from a cursor.
	schemaPositions map[string][]int
}

func newArtifactFacet() artifactFacet {
	return artifactFacet{
		records: map[artifact.ID]*artifactRecord{}, schemaPositions: map[string][]int{},
		byMedia: map[string][]artifact.ID{}, bySchema: map[string][]artifact.ID{},
		byKind: map[artifact.Kind][]artifact.ID{},
	}
}

func (f artifactFacet) has(id artifact.ID) bool {
	_, ok := f.records[id]
	return ok
}

func (f artifactFacet) count() int { return len(f.records) }

func (f artifactFacet) record(id artifact.ID) (*artifactRecord, bool) {
	record, ok := f.records[id]
	return record, ok
}

// bare reports a descriptor that names an identity and nothing else: what a
// batch declares for a document it references without carrying.
func bare(descriptor artifact.Descriptor) bool {
	return descriptor == artifact.Descriptor{ID: descriptor.ID}
}

// fills reports an incoming descriptor that completes a bare record: it
// states the facts the record lacks and the same batch carries the content
// they describe, which the content facet holds to the identity. It is the
// one way a declared identity gains its facts after it was first named.
func fills(existing, incoming artifact.Descriptor, batch artifact.Batch) bool {
	return bare(existing) && !bare(incoming) &&
		slices.ContainsFunc(batch.Contents, func(content artifact.Content) bool { return content.Descriptor == incoming })
}

func (f artifactFacet) validate(batch artifact.Batch) (map[artifact.ID]struct{}, error) {
	added := make(map[artifact.ID]struct{}, len(batch.Artifacts))
	for _, descriptor := range batch.Artifacts {
		// A bare declaration of a known identity adds nothing; a full one
		// fills a bare record only beside its content.
		if record, ok := f.records[descriptor.ID]; ok && record.descriptor != descriptor && !bare(descriptor) && !fills(record.descriptor, descriptor, batch) {
			return nil, fmt.Errorf("%w: %s", ErrArtifactConflict, descriptor.ID)
		}
		added[descriptor.ID] = struct{}{}
	}
	for _, manifest := range batch.Manifests {
		if record, ok := f.records[manifest.ID]; ok && record.hasManifest && !sameManifest(record.manifest, manifest) {
			return nil, fmt.Errorf("%w: manifest %s", ErrArtifactConflict, manifest.ID)
		}
	}
	return added, nil
}

func (f *artifactFacet) add(descriptor artifact.Descriptor, sequence uint64) {
	if record, found := f.records[descriptor.ID]; found {
		// A fill keeps the record's first sequence and gains its facts.
		if bare(record.descriptor) && !bare(descriptor) {
			record.descriptor = descriptor
			indexDescriptor(f.byMedia, descriptor.MediaType, descriptor.ID)
			indexDescriptor(f.bySchema, descriptor.Schema, descriptor.ID)
			f.indexSchemaPosition(descriptor.Schema, record.position)
		}
		return
	}
	f.records[descriptor.ID] = &artifactRecord{descriptor: descriptor, sequence: sequence, position: len(f.bySequence)}
	f.indexSchemaPosition(descriptor.Schema, len(f.bySequence))
	indexDescriptor(f.byMedia, descriptor.MediaType, descriptor.ID)
	indexDescriptor(f.bySchema, descriptor.Schema, descriptor.ID)
	f.byKind[descriptor.ID.Kind()] = append(f.byKind[descriptor.ID.Kind()], descriptor.ID)
	f.bySequence = append(f.bySequence, descriptor.ID)
}

// indexSchemaPosition files a record's position under its schema in
// ascending order; a new record appends, a filled one slots in.
func (f *artifactFacet) indexSchemaPosition(schema string, position int) {
	if schema == "" {
		return
	}
	positions := f.schemaPositions[schema]
	if at, found := slices.BinarySearch(positions, position); !found {
		f.schemaPositions[schema] = slices.Insert(positions, at, position)
	}
}

func (f *artifactFacet) setManifest(manifest artifact.Manifest) {
	record := f.records[manifest.ID]
	record.manifest, record.hasManifest = manifest.Clone(), true
}

// accept has no conditions beyond aggregate validation: descriptor
// and manifest conflicts were refused by validate.
func (f *artifactFacet) accept(artifact.Batch, map[artifact.ID]contentLocator, uint64) error {
	return nil
}

// applyCommit registers new descriptors and manifests in commit order.
func (f *artifactFacet) applyCommit(delta artifact.Batch, _ map[artifact.ID]contentLocator, sequence uint64) {
	for _, descriptor := range delta.Artifacts {
		f.add(descriptor, sequence)
	}
	for _, manifest := range delta.Manifests {
		f.setManifest(manifest)
	}
}

func (f artifactFacet) delta(batch artifact.Batch, delta *artifact.Batch) {
	for _, descriptor := range batch.Artifacts {
		// A fill is journaled like a new descriptor, so replay regains it.
		if record, found := f.records[descriptor.ID]; !found || bare(record.descriptor) && !bare(descriptor) {
			delta.Artifacts = append(delta.Artifacts, descriptor)
		}
	}
	for _, manifest := range batch.Manifests {
		if record, exists := f.records[manifest.ID]; !exists || !record.hasManifest {
			delta.Manifests = append(delta.Manifests, manifest)
		}
	}
}

// contentFacet owns the journal locator of every stored content blob,
// released ones included: a released locator keeps the introduction
// sequence and records the release sequence.
type contentFacet struct {
	locators map[artifact.ID]contentLocator
}

func newContentFacet() contentFacet {
	return contentFacet{locators: map[artifact.ID]contentLocator{}}
}

// has reports durable, unreleased bytes.
func (f contentFacet) has(id artifact.ID) bool {
	locator, ok := f.locators[id]
	return ok && locator.released == 0
}

// locator returns the recorded locator whether or not its bytes were
// released; callers that need bytes check released.
func (f contentFacet) locator(id artifact.ID) (contentLocator, bool) {
	locator, ok := f.locators[id]
	return locator, ok
}

func (f *contentFacet) set(id artifact.ID, locator contentLocator, sequence uint64) {
	locator.sequence = sequence
	f.locators[id] = locator
}

// accept refuses a commit whose content lacks a bound locator: bytes
// that never became durable must not gain a committed descriptor. A
// release must name bytes that are durable at this sequence.
func (f *contentFacet) accept(delta artifact.Batch, locators map[artifact.ID]contentLocator, sequence uint64) error {
	if (len(delta.Contents) != 0 || len(delta.Releases) != 0) && sequence == 0 {
		return errors.New("overgodb: durable content requires a commit sequence")
	}
	for _, content := range delta.Contents {
		if _, bound := locators[content.Descriptor.ID]; !bound {
			return fmt.Errorf("overgodb: content %s has no durable locator", content.Descriptor.ID)
		}
	}
	for _, release := range delta.Releases {
		if !f.has(release) {
			return fmt.Errorf("overgodb: release names content that is not durable: %s", release)
		}
	}
	return nil
}

// applyCommit binds each committed content to its locator and stamps
// each release on the locator it names.
func (f *contentFacet) applyCommit(delta artifact.Batch, locators map[artifact.ID]contentLocator, sequence uint64) {
	for _, content := range delta.Contents {
		f.set(content.Descriptor.ID, locators[content.Descriptor.ID], sequence)
	}
	for _, release := range delta.Releases {
		locator := f.locators[release]
		locator.released = sequence
		f.locators[release] = locator
	}
}

func (f contentFacet) delta(batch artifact.Batch, delta *artifact.Batch) {
	for _, content := range batch.Contents {
		if !f.has(content.Descriptor.ID) {
			delta.Contents = append(delta.Contents, content)
		}
	}
	for _, release := range batch.Releases {
		if f.has(release) {
			delta.Releases = append(delta.Releases, release)
		}
	}
}

// lineageFacet owns the acyclic dependency graph: both adjacency
// directions, the edge count, and cycle refusal.
type lineageFacet struct {
	parents  map[artifact.ID][]relationKey
	children map[artifact.ID][]relationKey
	edges    int
}

func newLineageFacet() lineageFacet {
	return lineageFacet{parents: map[artifact.ID][]relationKey{}, children: map[artifact.ID][]relationKey{}}
}

func (f lineageFacet) has(key relationKey) bool {
	_, found := slices.BinarySearchFunc(f.parents[key.child], key, compareRelation)
	return found
}

func (f lineageFacet) parentsOf(id artifact.ID) []relationKey  { return f.parents[id] }
func (f lineageFacet) childrenOf(id artifact.ID) []relationKey { return f.children[id] }
func (f lineageFacet) count() int                              { return f.edges }

func (f lineageFacet) validate(batch artifact.Batch, hasArtifact func(artifact.ID) bool) error {
	pendingParents := map[artifact.ID]map[artifact.ID]struct{}{}
	for _, edge := range batch.Lineage {
		if !hasArtifact(edge.Child) {
			return fmt.Errorf("overgodb: lineage child is unknown: %s", edge.Child)
		}
		if !hasArtifact(edge.Parent) {
			return fmt.Errorf("overgodb: lineage parent is unknown: %s", edge.Parent)
		}
		key := relationKey{child: edge.Child, parent: edge.Parent, relation: edge.Relation}
		if f.has(key) {
			continue
		}
		if f.reaches(edge.Parent, edge.Child, pendingParents) {
			return fmt.Errorf("%w: %s -> %s", ErrLineageCycle, edge.Child, edge.Parent)
		}
		parents := pendingParents[edge.Child]
		if parents == nil {
			parents = map[artifact.ID]struct{}{}
			pendingParents[edge.Child] = parents
		}
		parents[edge.Parent] = struct{}{}
	}
	return nil
}

func (f *lineageFacet) add(key relationKey) {
	if f.has(key) {
		return
	}
	f.parents[key.child] = insertRelation(f.parents[key.child], key)
	f.children[key.parent] = insertRelation(f.children[key.parent], key)
	f.edges++
}

// accept has no conditions beyond aggregate validation: cycles and
// unknown endpoints were refused by validate.
func (f *lineageFacet) accept(artifact.Batch, map[artifact.ID]contentLocator, uint64) error {
	return nil
}

// applyCommit inserts each committed edge into both adjacencies.
func (f *lineageFacet) applyCommit(delta artifact.Batch, _ map[artifact.ID]contentLocator, _ uint64) {
	for _, edge := range delta.Lineage {
		f.add(relationKey{child: edge.Child, parent: edge.Parent, relation: edge.Relation})
	}
}

func (f lineageFacet) delta(batch artifact.Batch, delta *artifact.Batch) {
	for _, edge := range batch.Lineage {
		if !f.has(relationKey{child: edge.Child, parent: edge.Parent, relation: edge.Relation}) {
			delta.Lineage = append(delta.Lineage, edge)
		}
	}
}

func (f lineageFacet) reaches(start, target artifact.ID, pending map[artifact.ID]map[artifact.ID]struct{}) bool {
	if start == target {
		return true
	}
	seen := map[artifact.ID]struct{}{start: {}}
	queue := []artifact.ID{start}
	visit := func(parent artifact.ID) bool {
		if parent == target {
			return true
		}
		if _, ok := seen[parent]; ok {
			return false
		}
		seen[parent] = struct{}{}
		queue = append(queue, parent)
		return false
	}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		for _, key := range f.parents[current] {
			if visit(key.parent) {
				return true
			}
		}
		for parent := range pending[current] {
			if visit(parent) {
				return true
			}
		}
	}
	return false
}

// locationFacet owns the sorted physical-location events per artifact.
type locationFacet struct {
	byArtifact map[artifact.ID][]artifact.Location
}

func newLocationFacet() locationFacet {
	return locationFacet{byArtifact: map[artifact.ID][]artifact.Location{}}
}

func (f locationFacet) of(id artifact.ID) []artifact.Location { return f.byArtifact[id] }

func (f locationFacet) has(location artifact.Location) bool {
	_, found := slices.BinarySearchFunc(f.byArtifact[location.Artifact], location, compareLocation)
	return found
}

func (f locationFacet) validate(batch artifact.Batch, hasArtifact func(artifact.ID) bool) error {
	for _, event := range batch.Locations {
		if !hasArtifact(event.Artifact) {
			return fmt.Errorf("overgodb: location artifact is unknown: %s", event.Artifact)
		}
		if event.Action == artifact.LocationRemove && !f.has(event.Location) {
			return fmt.Errorf("overgodb: remove unknown location %q", event.Value)
		}
	}
	return nil
}

func (f *locationFacet) apply(event artifact.LocationEvent) {
	locations := f.byArtifact[event.Artifact]
	index, found := slices.BinarySearchFunc(locations, event.Location, compareLocation)
	if event.Action == artifact.LocationAdd {
		if !found {
			f.byArtifact[event.Artifact] = slices.Insert(locations, index, event.Location)
		}
	} else if found {
		f.byArtifact[event.Artifact] = slices.Delete(locations, index, index+1)
	}
}

// accept has no conditions beyond aggregate validation.
func (f *locationFacet) accept(artifact.Batch, map[artifact.ID]contentLocator, uint64) error {
	return nil
}

// applyCommit replays each committed location event in order.
func (f *locationFacet) applyCommit(delta artifact.Batch, _ map[artifact.ID]contentLocator, _ uint64) {
	for _, event := range delta.Locations {
		f.apply(event)
	}
}

func (f locationFacet) delta(batch artifact.Batch, delta *artifact.Batch) {
	for _, event := range batch.Locations {
		if event.Action == artifact.LocationRemove || !f.has(event.Location) {
			delta.Locations = append(delta.Locations, event)
		}
	}
}

// aliasFacet owns mutable name bindings and their compare-and-set rule.
type aliasFacet struct {
	bindings map[string]artifact.ID
}

func newAliasFacet() aliasFacet { return aliasFacet{bindings: map[string]artifact.ID{}} }

func (f aliasFacet) resolve(name string) (artifact.ID, bool) {
	target, ok := f.bindings[name]
	return target, ok
}

func (f aliasFacet) count() int { return len(f.bindings) }

// each visits every binding in map order; callers own ordering.
func (f aliasFacet) each(visit func(name string, target artifact.ID)) {
	for name, target := range f.bindings {
		visit(name, target)
	}
}

func (f aliasFacet) validate(batch artifact.Batch, hasArtifact func(artifact.ID) bool) error {
	for _, binding := range batch.Aliases {
		if binding.Remove && strings.HasPrefix(binding.Name, StoreLocalAliasPrefix) {
			return fmt.Errorf("overgodb: store-local alias %q cannot be retired", binding.Name)
		}
		if !binding.Remove && !hasArtifact(binding.Target) {
			return fmt.Errorf("overgodb: alias %q targets unknown artifact %s", binding.Name, binding.Target)
		}
		current, exists := f.bindings[binding.Name]
		if binding.Previous == nil && exists {
			return fmt.Errorf("%w: %q is already bound", ErrAliasConflict, binding.Name)
		}
		if binding.Previous != nil && (!exists || current != *binding.Previous) {
			return fmt.Errorf("%w: %q has unexpected target", ErrAliasConflict, binding.Name)
		}
	}
	return nil
}

func (f *aliasFacet) apply(binding artifact.AliasBinding) {
	if binding.Remove {
		delete(f.bindings, binding.Name)
	} else {
		f.bindings[binding.Name] = binding.Target
	}
}

// accept has no conditions beyond aggregate validation: compare-and-
// set conflicts were refused by validate.
func (f *aliasFacet) accept(artifact.Batch, map[artifact.ID]contentLocator, uint64) error { return nil }

// applyCommit applies each committed binding in order.
func (f *aliasFacet) applyCommit(delta artifact.Batch, _ map[artifact.ID]contentLocator, _ uint64) {
	for _, binding := range delta.Aliases {
		f.apply(binding)
	}
}

func (f aliasFacet) delta(batch artifact.Batch, delta *artifact.Batch) {
	for _, binding := range batch.Aliases {
		current, exists := f.bindings[binding.Name]
		if binding.Remove || !exists || current != binding.Target {
			delta.Aliases = append(delta.Aliases, binding)
		}
	}
}

// commitFacet owns the ordered committed-batch record and its key index.
type commitFacet struct {
	ordered []committedBatch
	byKey   map[string]int
}

func newCommitFacet() commitFacet { return commitFacet{byKey: map[string]int{}} }

func (f commitFacet) lookup(key string) (committedBatch, bool) {
	index, found := f.byKey[key]
	if !found {
		return committedBatch{}, false
	}
	return f.ordered[index], true
}

func (f commitFacet) count() int                  { return len(f.ordered) }
func (f commitFacet) all() []committedBatch       { return f.ordered }
func (f commitFacet) at(index int) committedBatch { return f.ordered[index] }

func (f *commitFacet) add(commit committedBatch) {
	f.byKey[commit.key] = len(f.ordered)
	f.ordered = append(f.ordered, commit)
}

// accept has no conditions: the commit record is coordinator-owned.
func (f *commitFacet) accept(artifact.Batch, map[artifact.ID]contentLocator, uint64) error {
	return nil
}

// applyCommit is a no-op on the delta: the commit record advances
// through addCommit with the frame identity the coordinator owns.
func (f *commitFacet) applyCommit(artifact.Batch, map[artifact.ID]contentLocator, uint64) {}

func compareRelation(left, right relationKey) int {
	if order := artifact.CompareID(left.child, right.child); order != 0 {
		return order
	}
	if order := artifact.CompareID(left.parent, right.parent); order != 0 {
		return order
	}
	return cmp.Compare(left.relation, right.relation)
}

func insertRelation(edges []relationKey, key relationKey) []relationKey {
	index, found := slices.BinarySearchFunc(edges, key, compareRelation)
	if found {
		return edges
	}
	return slices.Insert(edges, index, key)
}

func compareLocation(left, right artifact.Location) int {
	if order := cmp.Compare(left.Kind, right.Kind); order != 0 {
		return order
	}
	return cmp.Compare(left.Value, right.Value)
}

// indexDescriptor files id under key. A record is filed once, when it first
// states the key, and a query orders what it selects itself, so the index
// appends: kept sorted, restoring a checkpoint moved each index once per
// record it held.
func indexDescriptor(index map[string][]artifact.ID, key string, id artifact.ID) {
	if key != "" {
		index[key] = append(index[key], id)
	}
}

func sameManifest(left, right artifact.Manifest) bool {
	return left.Version == right.Version && left.ID == right.ID && slices.Equal(left.Components, right.Components)
}
