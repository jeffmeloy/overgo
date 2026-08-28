package overgodb

import (
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
)

// StorageClass names where one kind of state may live. The classes are
// ownership rules, not suggestions: canonical facts are the only state
// that advances the artifact head, blobs carry immutable bytes those
// facts reference, projections are rebuildable acceleration, and
// operational signals stay bounded outside the store entirely unless
// they cause a durable transition.
type StorageClass string

const (
	// ClassCanonicalFact is journal-committed transaction state:
	// attempts, decisions, durable transitions, and evidence.
	ClassCanonicalFact StorageClass = "canonical-fact"
	// ClassImmutableBlob is content-addressed bytes referenced by
	// canonical facts: transcripts, observation chunks, tensors, media.
	ClassImmutableBlob StorageClass = "immutable-blob"
	// ClassRebuildableProjection is derived acceleration state --
	// checkpoints, snapshots, indexes -- reproducible from canonical
	// commits and never authority.
	ClassRebuildableProjection StorageClass = "rebuildable-projection"
	// ClassOperationalSignal is bounded coordination state --
	// heartbeats, wakeups, polling, partial process output -- that
	// must not become canonical unless it causes a durable transition.
	ClassOperationalSignal StorageClass = "operational-signal"
)

// operationalSignalMarkers name the coordination shapes whose schemas
// may never commit as canonical facts; the transition boundary refuses
// them. A signal that causes a durable transition is recorded through
// a typed transition document, never as the raw signal.
var operationalSignalMarkers = []string{
	"heartbeat", "wakeup", "poll", "partial-output", "liveness",
}

// DocumentStorageClass derives, from the descriptor alone, where an
// artifact's state belongs: a typed document is a canonical fact whose
// bytes ride the blob layout; schemaless raw bytes are immutable blob
// content referenced by facts.
func DocumentStorageClass(descriptor artifact.Descriptor) StorageClass {
	if IsOperationalSignalSchema(descriptor.Schema) {
		return ClassOperationalSignal
	}
	if descriptor.Schema != "" {
		return ClassCanonicalFact
	}
	return ClassImmutableBlob
}

// IsOperationalSignalSchema reports a schema that names bounded
// coordination rather than a durable fact.
func IsOperationalSignalSchema(schema string) bool {
	lowered := strings.ToLower(schema)
	for _, marker := range operationalSignalMarkers {
		if strings.Contains(lowered, marker) {
			return true
		}
	}
	return false
}

// LayoutStorageClass classifies one store-owned path by the layout
// rules: the journal and sealed segments carry canonical facts, the
// blob tree carries immutable bytes, and checkpoints and snapshots are
// rebuildable projections. Paths outside the layout are operational.
func LayoutStorageClass(path string) StorageClass {
	cleaned := filepath.ToSlash(path)
	switch {
	case strings.HasSuffix(cleaned, storeFilename),
		strings.Contains(cleaned, "/"+segmentDirectory+"/"):
		return ClassCanonicalFact
	case strings.Contains(cleaned, "/"+blobDirectory+"/"):
		return ClassImmutableBlob
	case strings.Contains(cleaned, "/"+checkpointDirectory+"/"),
		strings.Contains(cleaned, "/"+snapshotDirectory+"/"):
		return ClassRebuildableProjection
	}
	return ClassOperationalSignal
}
