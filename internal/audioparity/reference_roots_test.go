package audioparity

import (
	"cmp"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/repoanalysis"
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
func audioSources(t *testing.T, root string) repoanalysis.SourceSnapshot {
	t.Helper()
	paths := map[string]bool{}
	for _, patterns := range [][]string{
		{"-deps", "-test", "./internal/audioparity"},
		{"-deps", "./cmd/evaluate", "./cmd/recipe"},
	} {
		selection, err := repoanalysis.HostBuildSelection(root, patterns...)
		if err != nil {
			t.Fatal(err)
		}
		for path, selected := range selection.Files {
			if selected {
				paths[path] = true
			}
		}
	}
	source, err := repoanalysis.LoadGo(root, slices.Sorted(maps.Keys(paths)))
	if err != nil {
		t.Fatal(err)
	}
	return source
}

func resolveReferenceRoots(t *testing.T) referenceRoots {
	t.Helper()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	store := cmp.Or(os.Getenv("OVERGO_AUDIO_REFERENCE_STORE"), roots.AudioReference)
	return referenceRoots{store: store, datasets: roots.Datasets, models: filepath.Join(filepath.Dir(store), "models")}
}

// librispeechPath locates one LibriSpeech clean shard under the datasets root.
func librispeechPath(t *testing.T, elements ...string) string {
	t.Helper()
	return filepath.Join(append([]string{resolveReferenceRoots(t).datasets, "librispeech_asr-clean-xet"}, elements...)...)
}
