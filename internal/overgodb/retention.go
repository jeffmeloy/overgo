package overgodb

import (
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"

	"overgo/internal/artifact"
)

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
}

// Compact writes alias-rooted parent closure to a new store.
func Compact(ctx context.Context, source *Store, destinationRoot string) (RetentionReport, error) {
	report := RetentionReport{}
	if source == nil {
		return report, errors.New("overgodb retention: nil source store")
	}
	descriptors := map[artifact.ID]artifact.Descriptor{}
	manifests := map[artifact.ID]artifact.Manifest{}
	contents := map[artifact.ID]bool{}
	parents := map[artifact.ID][]relationKey{}
	locations := map[artifact.ID][]artifact.Location{}
	causality := map[artifact.ID]artifact.CausalLink{}
	source.mu.RLock()
	if err := source.ready(false); err != nil {
		source.mu.RUnlock()
		return report, err
	}
	for execution, link := range source.state.causality.records {
		causality[execution] = link.Clone()
	}
	for id, record := range source.state.artifacts.records {
		descriptors[id] = record.descriptor
		parents[id] = slices.Clone(source.state.lineage.parentsOf(id))
		locations[id] = slices.Clone(source.state.locations.of(id))
		if source.state.contents.has(id) {
			contents[id] = true
		}
		if record.hasManifest {
			manifests[id] = record.manifest.Clone()
		}
	}
	aliases := source.state.aliasViews("")
	source.mu.RUnlock()

	retained := map[artifact.ID]bool{}
	var retain func(artifact.ID)
	retain = func(id artifact.ID) {
		if retained[id] {
			return
		}
		if _, known := descriptors[id]; !known {
			return
		}
		retained[id] = true
		if manifest, ok := manifests[id]; ok {
			for _, component := range manifest.Components {
				retain(component.Artifact)
			}
		}
		for _, edge := range parents[id] {
			retain(edge.parent)
		}
		if link, found := causality[id]; found {
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

type compactionWriter struct {
	ctx         context.Context
	destination *Store
	items       []artifact.Batch
	content     int
	chunk       int
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
	fits, err := writer.destination.transactionFits(batch)
	if err != nil {
		return err
	}
	if fits {
		if _, err := writer.destination.Commit(writer.ctx, batch); err != nil {
			return err
		}
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
	return fmt.Sprintf("retention/compact/%d", writer.chunk+1)
}

func mergeCompactionItems(items []artifact.Batch) artifact.Batch {
	var artifacts, contents, manifests, lineage, causality, aliases, locations int
	for _, item := range items {
		artifacts += len(item.Artifacts)
		contents += len(item.Contents)
		manifests += len(item.Manifests)
		lineage += len(item.Lineage)
		causality += len(item.Causality)
		aliases += len(item.Aliases)
		locations += len(item.Locations)
	}
	merged := artifact.Batch{
		Artifacts: make([]artifact.Descriptor, artifacts),
		Contents:  make([]artifact.Content, contents),
		Manifests: make([]artifact.Manifest, manifests),
		Lineage:   make([]artifact.Lineage, lineage),
		Causality: make([]artifact.CausalLink, causality),
		Aliases:   make([]artifact.AliasBinding, aliases),
		Locations: make([]artifact.LocationEvent, locations),
	}
	var artifactAt, contentAt, manifestAt, lineageAt, causalityAt, aliasAt, locationAt int
	for _, item := range items {
		artifactAt += copy(merged.Artifacts[artifactAt:], item.Artifacts)
		contentAt += copy(merged.Contents[contentAt:], item.Contents)
		manifestAt += copy(merged.Manifests[manifestAt:], item.Manifests)
		lineageAt += copy(merged.Lineage[lineageAt:], item.Lineage)
		causalityAt += copy(merged.Causality[causalityAt:], item.Causality)
		aliasAt += copy(merged.Aliases[aliasAt:], item.Aliases)
		locationAt += copy(merged.Locations[locationAt:], item.Locations)
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
