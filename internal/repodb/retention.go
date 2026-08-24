package repodb

import (
	"context"
	"errors"
	"fmt"
	"io"
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
	DroppedArtifacts  int
}

// Compact writes alias-rooted parent closure to a new store.
func Compact(ctx context.Context, source *Store, destinationRoot string) (RetentionReport, error) {
	report := RetentionReport{}
	if source == nil {
		return report, errors.New("repodb retention: nil source store")
	}
	descriptors := map[artifact.ID]artifact.Descriptor{}
	manifests := map[artifact.ID]artifact.Manifest{}
	contents := map[artifact.ID]bool{}
	parents := map[artifact.ID][]relationKey{}
	locations := map[artifact.ID][]artifact.Location{}
	source.mu.RLock()
	if err := source.ready(false); err != nil {
		source.mu.RUnlock()
		return report, err
	}
	for id, slot := range source.state.slots {
		descriptors[id] = slot.descriptor
		parents[id] = slices.Clone(slot.parents)
		locations[id] = slices.Clone(slot.locations)
		if slot.hasContent {
			contents[id] = true
		}
		if slot.hasManifest {
			manifests[id] = slot.manifest.Clone()
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
	}
	for _, alias := range aliases {
		retain(alias.Target)
	}
	ids := make([]artifact.ID, 0, len(retained))
	for id := range retained {
		ids = append(ids, id)
	}
	slices.SortFunc(ids, artifact.CompareID)

	destination, err := Open(destinationRoot)
	if err != nil {
		return report, err
	}
	defer destination.Close()
	if _, sequence := destination.Head(); sequence != 0 {
		return report, fmt.Errorf("repodb retention: destination %s already holds %d commit(s)", destinationRoot, sequence)
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
	return report, nil
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
		return errors.New("repodb retention: one compacted fact exceeds the frame limit")
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
		return errors.New("repodb retention: one compacted fact exceeds the frame limit")
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
	var artifacts, contents, manifests, lineage, aliases, locations int
	for _, item := range items {
		artifacts += len(item.Artifacts)
		contents += len(item.Contents)
		manifests += len(item.Manifests)
		lineage += len(item.Lineage)
		aliases += len(item.Aliases)
		locations += len(item.Locations)
	}
	merged := artifact.Batch{
		Artifacts: make([]artifact.Descriptor, artifacts),
		Contents:  make([]artifact.Content, contents),
		Manifests: make([]artifact.Manifest, manifests),
		Lineage:   make([]artifact.Lineage, lineage),
		Aliases:   make([]artifact.AliasBinding, aliases),
		Locations: make([]artifact.LocationEvent, locations),
	}
	var artifactAt, contentAt, manifestAt, lineageAt, aliasAt, locationAt int
	for _, item := range items {
		artifactAt += copy(merged.Artifacts[artifactAt:], item.Artifacts)
		contentAt += copy(merged.Contents[contentAt:], item.Contents)
		manifestAt += copy(merged.Manifests[manifestAt:], item.Manifests)
		lineageAt += copy(merged.Lineage[lineageAt:], item.Lineage)
		aliasAt += copy(merged.Aliases[aliasAt:], item.Aliases)
		locationAt += copy(merged.Locations[locationAt:], item.Locations)
	}
	return merged
}

// String renders the report.
func (report RetentionReport) String() string {
	return fmt.Sprintf("retained artifacts=%d manifests=%d contents=%d aliases=%d lineage=%d; dropped=%d",
		report.RetainedArtifacts, report.RetainedManifests, report.RetainedContents,
		report.Aliases, report.Lineage, report.DroppedArtifacts)
}
