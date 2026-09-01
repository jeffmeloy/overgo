package overgodb

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sort"

	"overgo/internal/artifact"
)

// RebuildReport states what a rebuild carried and what it stripped, so
// the operation is auditable from its own output.
type RebuildReport struct {
	Artifacts       int
	ContentsKept    int
	ContentsDropped int
	BytesKept       uint64
	BytesDropped    uint64
	Manifests       int
	LineageEdges    int
	CausalLinks     int
	Locations       int
	Aliases         int
	Batches         int
}

// rebuildBatchBytes bounds one rebuild batch comfortably inside the
// frame payload limit so a large kept content never overflows a frame.
const (
	rebuildBatchBytes        = 16 << 20
	rebuildManifestChunkSize = 256
	rebuildGraphChunkSize    = 2048
	rebuildAliasChunkSize    = 512
)

// Rebuild writes the source store's projected state into a fresh store
// at destination, in original creation order so every dependency
// precedes its dependents. strip decides whose CONTENT is dropped; a
// stripped artifact keeps its registered identity, size, lineage, and
// locations, so references to it stay resolvable. The source is not
// modified.
func Rebuild(ctx context.Context, source *Store, destination string, strip func(artifact.Descriptor) bool, admit StoreLocalAdmission) (RebuildReport, error) {
	if source == nil || destination == "" {
		return RebuildReport{}, errors.New("overgodb: rebuild needs a source store and destination")
	}
	if strip == nil {
		strip = func(artifact.Descriptor) bool { return false }
	}
	source.mu.RLock()
	if err := source.ready(false); err != nil {
		source.mu.RUnlock()
		return RebuildReport{}, err
	}
	if err := admitStoreLocalAliases("rebuild", source.state.aliasViews(""), admit); err != nil {
		source.mu.RUnlock()
		return RebuildReport{}, err
	}
	source.mu.RUnlock()
	target, err := Open(destination)
	if err != nil {
		return RebuildReport{}, err
	}
	defer target.Close()

	source.mu.RLock()
	order := append([]artifact.ID(nil), source.state.artifacts.bySequence...)
	causalLinks := make([]artifact.CausalLink, 0, len(source.state.causality.records))
	for _, link := range source.state.causality.records {
		causalLinks = append(causalLinks, link.Clone())
	}
	sort.Slice(causalLinks, func(i, j int) bool {
		return artifact.CompareID(causalLinks[i].Execution, causalLinks[j].Execution) < 0
	})
	source.mu.RUnlock()

	report := RebuildReport{}
	var manifests []artifact.Manifest
	var edges []relationKey
	batch := artifact.Batch{}
	var batchBytes uint64
	flush := func() error {
		if len(batch.Artifacts) == 0 && len(batch.Contents) == 0 && len(batch.Manifests) == 0 &&
			len(batch.Lineage) == 0 && len(batch.Causality) == 0 && len(batch.Locations) == 0 && len(batch.Aliases) == 0 {
			return nil
		}
		batch.Key = fmt.Sprintf("rebuild/%06d", report.Batches)
		if _, err := target.Commit(ctx, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
			return fmt.Errorf("overgodb: rebuild batch %d: %w", report.Batches, err)
		}
		report.Batches++
		batch = artifact.Batch{}
		batchBytes = uint64(len(batch.Contents))
		return nil
	}

	for _, id := range order {
		source.mu.RLock()
		record, ok := source.state.artifacts.record(id)
		if !ok {
			source.mu.RUnlock()
			continue
		}
		descriptor := record.descriptor
		hasManifest := record.manifest.ID.Valid()
		manifest := record.manifest
		parents := append([]relationKey(nil), source.state.lineage.parentsOf(id)...)
		locations := append([]artifact.Location(nil), source.state.locations.of(id)...)
		locator, hasContent := source.state.contents.locator(id)
		source.mu.RUnlock()

		report.Artifacts++
		batch.Artifacts = append(batch.Artifacts, descriptor)
		if hasContent {
			if strip(descriptor) {
				report.ContentsDropped++
				report.BytesDropped += descriptor.Size
			} else {
				data, err := source.materializeContent(id, locator)
				if err != nil {
					return report, fmt.Errorf("overgodb: rebuild content %s: %w", id, err)
				}
				batch.Contents = append(batch.Contents, artifact.Content{Descriptor: descriptor, Data: data})
				report.ContentsKept++
				report.BytesKept += descriptor.Size
				batchBytes += descriptor.Size
			}
		}
		if hasManifest {
			manifests = append(manifests, manifest)
		}
		edges = append(edges, parents...)
		for _, location := range locations {
			batch.Locations = append(batch.Locations, artifact.LocationEvent{
				Location: location, Action: artifact.LocationAdd,
			})
			report.Locations++
		}
		if batchBytes >= rebuildBatchBytes {
			if err := flush(); err != nil {
				return report, err
			}
		}
	}
	if err := flush(); err != nil {
		return report, err
	}
	// Manifests and lineage land after every artifact is registered:
	// an edge or component may join artifacts from different eras.
	for len(manifests) > 0 {
		chunk := manifests
		if len(chunk) > rebuildManifestChunkSize {
			chunk = manifests[:rebuildManifestChunkSize]
		}
		batch = artifact.Batch{Manifests: chunk}
		if err := flush(); err != nil {
			return report, err
		}
		report.Manifests += len(chunk)
		manifests = manifests[len(chunk):]
	}
	for len(edges) > 0 {
		chunk := edges
		if len(chunk) > rebuildGraphChunkSize {
			chunk = edges[:rebuildGraphChunkSize]
		}
		lineage := make([]artifact.Lineage, len(chunk))
		for index, edge := range chunk {
			lineage[index] = artifact.Lineage{Child: edge.child, Parent: edge.parent, Relation: edge.relation}
		}
		batch = artifact.Batch{Lineage: lineage}
		if err := flush(); err != nil {
			return report, err
		}
		report.LineageEdges += len(chunk)
		edges = edges[len(chunk):]
	}
	for len(causalLinks) > 0 {
		chunk := causalLinks
		if len(chunk) > rebuildGraphChunkSize {
			chunk = causalLinks[:rebuildGraphChunkSize]
		}
		batch = artifact.Batch{Causality: chunk}
		if err := flush(); err != nil {
			return report, err
		}
		report.CausalLinks += len(chunk)
		causalLinks = causalLinks[len(chunk):]
	}

	source.mu.RLock()
	names := make([]string, 0, source.state.aliases.count())
	source.state.aliases.each(func(name string, _ artifact.ID) {
		names = append(names, name)
	})
	slices.Sort(names)
	bindings := make([]artifact.AliasBinding, 0, len(names))
	for _, name := range names {
		target, _ := source.state.aliases.resolve(name)
		bindings = append(bindings, artifact.AliasBinding{Name: name, Target: target})
	}
	source.mu.RUnlock()
	for len(bindings) > 0 {
		chunk := bindings
		if len(chunk) > rebuildAliasChunkSize {
			chunk = bindings[:rebuildAliasChunkSize]
		}
		batch = artifact.Batch{Aliases: chunk}
		if err := flush(); err != nil {
			return report, err
		}
		report.Aliases += len(chunk)
		bindings = bindings[len(chunk):]
	}
	return report, nil
}
