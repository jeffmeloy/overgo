package overgodb

import (
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
)

// TestRSIStorageClassContract holds the ownership rules to their
// definitions: typed documents are canonical facts, schemaless bytes
// are immutable blobs, coordination-shaped schemas are operational
// signals, and every path in the store layout classifies to exactly
// the class its directory owns.
func TestRSIStorageClassContract(t *testing.T) {
	document := artifact.Descriptor{Schema: "overgo/attempt/v1", MediaType: "application/json"}
	if class := DocumentStorageClass(document); class != ClassCanonicalFact {
		t.Fatalf("typed document classifies %s", class)
	}
	raw := artifact.Descriptor{MediaType: "application/octet-stream"}
	if class := DocumentStorageClass(raw); class != ClassImmutableBlob {
		t.Fatalf("raw bytes classify %s", class)
	}
	for _, schema := range []string{
		"overgo/agent-heartbeat/v1", "overgo/session-wakeup/v1",
		"overgo/queue-poll/v1", "overgo/partial-output/v1", "overgo/liveness-probe/v1",
	} {
		signal := artifact.Descriptor{Schema: schema}
		if class := DocumentStorageClass(signal); class != ClassOperationalSignal {
			t.Fatalf("signal schema %q classifies %s", schema, class)
		}
	}
	layout := map[string]StorageClass{
		filepath.Join("root", storeFilename):                                      ClassCanonicalFact,
		filepath.Join("root", segmentDirectory, "x"+segmentExtension):             ClassCanonicalFact,
		filepath.Join("root", blobDirectory, "run", "ab", "abcd"):                 ClassImmutableBlob,
		filepath.Join("root", checkpointDirectory, "aliases"+checkpointExtension): ClassRebuildableProjection,
		filepath.Join("root", snapshotDirectory, "x"+snapshotExtension):           ClassRebuildableProjection,
		filepath.Join("root", "overgodb.lock"):                                    ClassOperationalSignal,
	}
	for path, expected := range layout {
		if class := LayoutStorageClass(path); class != expected {
			t.Fatalf("layout path %q classifies %s, want %s", path, class, expected)
		}
	}
}
