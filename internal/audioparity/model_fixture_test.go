package audioparity

import (
	"encoding/hex"
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/safetensors"
	"overgo/internal/testcheck"
	"path/filepath"
	"testing"
)

func loadASRModel(t *testing.T, store *overgodb.Store, election Election) (string, *safetensors.Source) {
	t.Helper()
	modelRoot, err := artifact.AvailablePath(t.Context(), store, election.Model.ID, artifact.LocationDirectory)
	testcheck.NoError(t, err)
	weights, err := safetensors.OpenSource(modelRoot)
	testcheck.NoError(t, err)
	t.Cleanup(func() { _ = weights.Close() })
	digests, err := weights.ShardDigests()
	testcheck.NoError(t, err)
	weightCount := 0
	for _, file := range election.Files {
		if file.Role == "weights" {
			matched := false
			for _, digest := range digests {
				if digest.Name == file.Path && hex.EncodeToString(digest.Digest[:]) == file.Artifact.DigestHex() && digest.Size == file.Bytes {
					matched = true
				}
			}
			if !matched {
				t.Fatalf("weight shard differs: %s", file.Path)
			}
			weightCount++
		} else if file.Role == "tokenizer" || file.Role == "configuration" || file.Role == "preprocessor" {
			verifyASRFile(t, filepath.Join(modelRoot, filepath.FromSlash(file.Path)), file.Artifact, file.Bytes)
		}
	}
	if weightCount == 0 || weightCount != len(digests) {
		t.Fatal("unexpected or missing weight shard")
	}
	return modelRoot, weights
}
