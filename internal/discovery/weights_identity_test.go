package discovery

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
)

// The weights identity is the model digest of the file's bytes, read
// through the memo on the second call, and a missing file is an error
// rather than an empty identity.
func TestWeightsIdentityDigestsThroughTheMemo(t *testing.T) {
	path := filepath.Join(t.TempDir(), "weights.gguf")
	if err := os.WriteFile(path, []byte("weights fixture bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	want, _, err := artifact.Identify(artifact.KindModel, file)
	file.Close()
	if err != nil {
		t.Fatal(err)
	}
	memo := NewMemo()
	got, err := WeightsIdentity(path, memo)
	if err != nil || got != want {
		t.Fatalf("identity = %s, %v; want %s", got, err, want)
	}
	if !memo.dirty || len(memo.entries) != 1 {
		t.Fatalf("memo did not record the digest: dirty=%v entries=%d", memo.dirty, len(memo.entries))
	}
	memo.dirty = false
	again, err := WeightsIdentity(path, memo)
	if err != nil || again != want || memo.dirty {
		t.Fatalf("second identity = %s, %v, dirty=%v; want the remembered %s", again, err, memo.dirty, want)
	}
	if _, err := WeightsIdentity(filepath.Join(t.TempDir(), "absent.gguf"), memo); err == nil {
		t.Fatal("an absent file must be an error")
	}
}
