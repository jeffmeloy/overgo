package main

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/testcheck"
)

// TestARequestNamingAModelByIdentityIsAnsweredByThatModel holds -init's half of loop-proof/proxy-model: the default
// worker names its model by a reference a fresh proxy's catalog resolves to that one model. On 2026-11-09 the catalog
// held two present entries of Qwen3.8-27B's file name and refused the file name as ambiguous; the model's identity
// resolves exactly, and the proxy hands the child its own name (internal/modelswap's test of the same name).
func TestARequestNamingAModelByIdentityIsAnsweredByThatModel(t *testing.T) {
	t.Parallel()
	worker := testcheck.Must(artifact.ParseID(defaultLocalModel))(t)
	other := testcheck.Must(artifact.ParseID("model:sha256:" + "00000000000000000000000000000000000000000000000000000000000000aa"))(t)
	entries := []discovery.CatalogEntry{
		{Model: worker, Location: "C:/models/Qwen3.8-27B-UD-Q6_K/Qwen3.8-27B-UD-Q6_K.gguf", Present: true},
		{Model: other, Location: "D:/copies/Qwen3.8-27B-UD-Q6_K.gguf", Present: true},
	}
	entry, found, err := discovery.MatchReference(entries, false, defaultLocalModel)
	if err != nil || !found || entry.Model != worker {
		t.Fatalf("-init's worker model %q resolves to %v (found %t, %v) in a catalog holding two entries of its file name; want the one model it names",
			defaultLocalModel, entry.Model, found, err)
	}
}
