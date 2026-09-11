package audioparity

import (
	"cmp"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
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
