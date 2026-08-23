package repodb

import (
	"context"
	"errors"
	"fmt"
	"regexp"

	"overgo/internal/artifact"
)

// Retention is compact-copy: the store is append-only and immutable, so
// nothing is ever deleted in place. Compact writes a new store holding the
// live set -- everything reachable from any alias through lineage edges and
// through artifact identities embedded in retained document content -- and
// the source store survives untouched as the archive of record.
//
// Reachability through content is deliberately conservative: any string in a
// retained document that parses as an artifact identity retains its target,
// so a lifecycle event's evidence array keeps its gate and run records even
// though no lineage row names them. Superseded closure decisions, whose
// active aliases moved past them, are exactly what this leaves behind.

// RetentionReport counts one compaction's outcome.
type RetentionReport struct {
	RetainedArtifacts int
	RetainedManifests int
	RetainedContents  int
	Aliases           int
	Lineage           int
	DroppedArtifacts  int
}

const (
	// compactChunkBytes bounds one compaction commit's payload so the live
	// set of any size lands as ordered chunks under the store's batch limit.
	compactChunkBytes = 8 << 20
	// compactEdgeCost approximates one lineage row's payload for chunking.
	compactEdgeCost = 256
)

var embeddedIdentity = regexp.MustCompile(`[a-z][a-z0-9-]*:sha256:[0-9a-f]{64}`)

// Compact writes the live set of source into a new store at destinationRoot,
// which must not already exist.
func Compact(ctx context.Context, source *Store, destinationRoot string) (RetentionReport, error) {
	report := RetentionReport{}
	if source == nil {
		return report, errors.New("repodb retention: nil source store")
	}
	descriptors := map[artifact.ID]artifact.Descriptor{}
	manifests := map[artifact.ID]artifact.Manifest{}
	source.mu.RLock()
	if err := source.ready(false); err != nil {
		source.mu.RUnlock()
		return report, err
	}
	for id, slot := range source.state.slots {
		descriptors[id] = slot.descriptor
		if slot.hasManifest {
			manifests[id] = slot.manifest.Clone()
		}
	}
	aliases := source.state.aliasViews("")
	source.mu.RUnlock()

	retained := map[artifact.ID]bool{}
	var walk func(id artifact.ID) error
	walk = func(id artifact.ID) error {
		if !id.Valid() || retained[id] {
			return nil
		}
		if _, known := descriptors[id]; !known {
			if _, known := manifests[id]; !known {
				return nil
			}
		}
		retained[id] = true
		if manifest, ok := manifests[id]; ok {
			for _, component := range manifest.Components {
				if err := walk(component.Artifact); err != nil {
					return err
				}
			}
		}
		parents, err := source.Parents(ctx, id)
		if err != nil {
			return err
		}
		children, err := source.Children(ctx, id)
		if err != nil {
			return err
		}
		for _, edge := range append(parents, children...) {
			if err := walk(edge.Parent); err != nil {
				return err
			}
			if err := walk(edge.Child); err != nil {
				return err
			}
		}
		content, ok, err := artifact.ReadContent(ctx, source, id)
		if err != nil {
			return err
		}
		if ok {
			for _, match := range embeddedIdentity.FindAll(content.Data, -1) {
				referenced, parseErr := artifact.ParseID(string(match))
				if parseErr != nil {
					continue
				}
				if err := walk(referenced); err != nil {
					return err
				}
			}
		}
		return nil
	}
	for _, alias := range aliases {
		if err := walk(alias.Target); err != nil {
			return report, err
		}
	}

	destination, err := Open(destinationRoot)
	if err != nil {
		return report, err
	}
	defer destination.Close()
	if _, sequence := destination.Head(); sequence != 0 {
		return report, fmt.Errorf("repodb retention: destination %s already holds %d commit(s)", destinationRoot, sequence)
	}
	// The live set can exceed one batch's payload limit, so compaction lands
	// as ordered chunks: artifacts and contents first, lineage among retained
	// artifacts second, aliases last, each chunk bounded by payload bytes.
	chunk := 0
	flush := func(batch *artifact.Batch, force bool, payload *int) error {
		if !force && *payload < compactChunkBytes {
			return nil
		}
		if len(batch.Artifacts)+len(batch.Contents)+len(batch.Manifests)+len(batch.Lineage)+len(batch.Aliases)+len(batch.Locations) == 0 {
			return nil
		}
		chunk++
		batch.Key = fmt.Sprintf("retention/compact/%d", chunk)
		if _, err := destination.Commit(ctx, *batch); err != nil {
			return err
		}
		*batch = artifact.Batch{}
		*payload = 0
		return nil
	}
	payload := 0
	batch := artifact.Batch{}
	// Phase one: every retained artifact and its content, so later phases can
	// only reference identities that already exist in the destination.
	for id := range retained {
		if _, ok := manifests[id]; ok {
			continue
		}
		if err := flush(&batch, false, &payload); err != nil {
			return report, err
		}
		descriptor := descriptors[id]
		content, ok, err := artifact.ReadContent(ctx, source, id)
		if err != nil {
			return report, err
		}
		if ok {
			batch.Contents = append(batch.Contents, content)
			payload += len(content.Data)
			report.RetainedContents++
		} else {
			batch.Artifacts = append(batch.Artifacts, descriptor)
			locations, err := source.Locations(ctx, id)
			if err != nil {
				return report, err
			}
			for _, location := range locations {
				batch.Locations = append(batch.Locations, artifact.LocationEvent{Location: location, Action: artifact.LocationAdd})
			}
		}
		report.RetainedArtifacts++
	}
	if err := flush(&batch, true, &payload); err != nil {
		return report, err
	}
	// Phase two: manifests, whose components all landed in phase one.
	for id := range retained {
		manifest, ok := manifests[id]
		if !ok {
			continue
		}
		batch.Manifests = append(batch.Manifests, manifest)
		payload += compactEdgeCost * len(manifest.Components)
		report.RetainedManifests++
		if err := flush(&batch, false, &payload); err != nil {
			return report, err
		}
	}
	if err := flush(&batch, true, &payload); err != nil {
		return report, err
	}
	seenEdges := map[string]bool{}
	for id := range retained {
		parents, err := source.Parents(ctx, id)
		if err != nil {
			return report, err
		}
		for _, edge := range parents {
			if !retained[edge.Parent] || !retained[edge.Child] {
				continue
			}
			key := edge.Child.String() + "\x00" + edge.Parent.String() + "\x00" + string(edge.Relation)
			if seenEdges[key] {
				continue
			}
			seenEdges[key] = true
			batch.Lineage = append(batch.Lineage, edge)
			payload += compactEdgeCost
			report.Lineage++
			if err := flush(&batch, false, &payload); err != nil {
				return report, err
			}
		}
	}
	if err := flush(&batch, true, &payload); err != nil {
		return report, err
	}
	for _, alias := range aliases {
		if !retained[alias.Target] {
			continue
		}
		batch.Aliases = append(batch.Aliases, artifact.AliasBinding{Name: alias.Name, Target: alias.Target})
		report.Aliases++
	}
	if err := flush(&batch, true, &payload); err != nil {
		return report, err
	}
	report.DroppedArtifacts = len(descriptors) + len(manifests) - report.RetainedArtifacts - report.RetainedManifests
	return report, nil
}

// String renders the report for operators.
func (report RetentionReport) String() string {
	return fmt.Sprintf("retained artifacts=%d manifests=%d contents=%d aliases=%d lineage=%d; dropped=%d",
		report.RetainedArtifacts, report.RetainedManifests, report.RetainedContents,
		report.Aliases, report.Lineage, report.DroppedArtifacts)
}
