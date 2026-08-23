package repodb

import (
	"context"
	"errors"
	"fmt"
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
	for _, id := range ids {
		if _, manifest := manifests[id]; manifest {
			continue
		}
		item := artifact.Batch{}
		content, found, err := artifact.ReadContent(ctx, source, id)
		if err != nil {
			return report, err
		}
		if found {
			item.Contents = []artifact.Content{content}
			report.RetainedContents++
		} else {
			item.Artifacts = []artifact.Descriptor{descriptors[id]}
		}
		for _, location := range locations[id] {
			item.Locations = append(item.Locations, artifact.LocationEvent{Location: location, Action: artifact.LocationAdd})
		}
		if err := writer.add(item); err != nil {
			return report, err
		}
		report.RetainedArtifacts++
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
	report.DroppedArtifacts = len(descriptors) - report.RetainedArtifacts - report.RetainedManifests
	return report, nil
}

type compactionWriter struct {
	ctx         context.Context
	destination *Store
	batch       artifact.Batch
	chunk       int
}

func (writer *compactionWriter) add(item artifact.Batch) error {
	previous := writer.batch
	writer.batch.Key = writer.nextKey()
	appendCompactionBatch(&writer.batch, item)
	fits, err := writer.destination.transactionFits(writer.batch)
	if err != nil {
		return err
	}
	if fits {
		return nil
	}
	writer.batch = previous
	if previous.Empty() {
		return errors.New("repodb retention: one compacted fact exceeds the frame limit")
	}
	if err := writer.flush(); err != nil {
		return err
	}
	writer.batch.Key = writer.nextKey()
	appendCompactionBatch(&writer.batch, item)
	fits, err = writer.destination.transactionFits(writer.batch)
	if err != nil {
		return err
	}
	if !fits {
		return errors.New("repodb retention: one compacted fact exceeds the frame limit")
	}
	return nil
}

func (writer *compactionWriter) flush() error {
	if writer.batch.Empty() {
		return nil
	}
	if _, err := writer.destination.Commit(writer.ctx, writer.batch); err != nil {
		return err
	}
	writer.chunk++
	writer.batch = artifact.Batch{}
	return nil
}

func (writer *compactionWriter) nextKey() string {
	return fmt.Sprintf("retention/compact/%d", writer.chunk+1)
}

func appendCompactionBatch(destination *artifact.Batch, source artifact.Batch) {
	destination.Artifacts = append(destination.Artifacts, source.Artifacts...)
	destination.Contents = append(destination.Contents, source.Contents...)
	destination.Manifests = append(destination.Manifests, source.Manifests...)
	destination.Lineage = append(destination.Lineage, source.Lineage...)
	destination.Aliases = append(destination.Aliases, source.Aliases...)
	destination.Locations = append(destination.Locations, source.Locations...)
}

// String renders the report.
func (report RetentionReport) String() string {
	return fmt.Sprintf("retained artifacts=%d manifests=%d contents=%d aliases=%d lineage=%d; dropped=%d",
		report.RetainedArtifacts, report.RetainedManifests, report.RetainedContents,
		report.Aliases, report.Lineage, report.DroppedArtifacts)
}
