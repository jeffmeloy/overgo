package audioparity

import (
	"cmp"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/gosource"
	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// referenceRoots resolves the acceptances' reference locations from the
// declared data roots: the reference store (the operator's override, else
// the store the roots declare for it), the datasets root, and the audio
// models beside the reference store. A checkout whose store lacks the
// reference records declares another checkout's in local-models.json; no
// operator environment is needed.
type referenceRoots struct {
	store, datasets, models string
}

// Bind repository Go sources compiled into the acceptance and its CLI producers.
// Models, corpus, native artifacts and protocol identities remain separate inputs.
func audioSources(t *testing.T, root string) gosource.Snapshot {
	t.Helper()
	paths := map[string]bool{}
	for _, patterns := range [][]string{
		{"-test", "./internal/audioparity"},
		{"./cmd/evaluate", "./cmd/recipe"},
	} {
		selection := testcheck.Must(gosource.HostBuildClosure(root, patterns...))(t)
		for path, selected := range selection.Files {
			if selected {
				paths[path] = true
			}
		}
	}
	source := testcheck.Must(gosource.LoadGo(root, slices.Sorted(maps.Keys(paths))))(t)
	return source
}

// The acceptances judge the recorded models, corpus and reference store, not
// a change: go test skips them unless the store acceptance is opted into,
// and the improvement cycle's audio lane opts in when a change reaches this
// package.
func resolveReferenceRoots(t *testing.T) referenceRoots {
	t.Helper()
	testskip.StoreAcceptance(t)
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	store := cmp.Or(os.Getenv("OVERGO_AUDIO_REFERENCE_STORE"), roots.AudioReference)
	return referenceRoots{store: store, datasets: roots.Datasets, models: filepath.Join(filepath.Dir(store), "models")}
}

// librispeechPath locates one LibriSpeech clean shard under the datasets root.
func librispeechPath(t *testing.T, elements ...string) string {
	t.Helper()
	return filepath.Join(append([]string{resolveReferenceRoots(t).datasets, "librispeech_asr-clean-xet"}, elements...)...)
}

// Resolve retained content independently of the checkout and verify its identity.
func retainedAudioFile(t *testing.T, digest string, size uint64) string {
	t.Helper()
	testskip.StoreAcceptance(t)
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	store := testcheck.Must(overgodb.OpenReadOnly(roots.Store))(t)
	defer store.Close()
	id := alignmentFileID(t, digest)
	path := testcheck.Must(artifact.AvailablePath(t.Context(), store, id, artifact.LocationFile))(t)
	verifyASRFile(t, path, id, size)
	return path
}
