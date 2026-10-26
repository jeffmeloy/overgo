package overgodb

import (
	"path/filepath"
	"testing"
)

// TestRSIStorageClassContract holds the ownership rules to their
// definitions: coordination-shaped schemas are operational signals,
// and every path in the store layout classifies to exactly the class
// its directory owns.
func TestRSIStorageClassContract(t *testing.T) {
	t.Parallel()
	for _, schema := range []string{
		"overgo/agent-heartbeat/v1", "overgo/session-wakeup/v1",
		"overgo/queue-poll/v1", "overgo/partial-output/v1", "overgo/liveness-probe/v1",
	} {
		if !IsOperationalSignalSchema(schema) {
			t.Fatalf("signal schema %q is not an operational signal", schema)
		}
	}
	layout := map[string]StorageClass{
		filepath.Join("root", storeFilename):                                      ClassCanonicalFact,
		filepath.Join("root", segmentDirectory, "x"+segmentExtension):             ClassCanonicalFact,
		filepath.Join("root", blobDirectory, "run", "ab", "abcd"):                 ClassImmutableBlob,
		filepath.Join("root", checkpointDirectory, "aliases"+checkpointExtension): ClassRebuildableProjection,
		filepath.Join("root", snapshotDirectory, "x.snapshot"):                    ClassRebuildableProjection,
		writerLockPath("root"):                                                    ClassOperationalSignal,
	}
	for path, expected := range layout {
		if class := LayoutStorageClass(path); class != expected {
			t.Fatalf("layout path %q classifies %s, want %s", path, class, expected)
		}
	}
}
