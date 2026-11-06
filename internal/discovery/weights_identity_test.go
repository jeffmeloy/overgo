package discovery

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

// The weights identity is the model digest of the file's bytes, read
// through the memo on the second call, and a missing file is an error
// rather than an empty identity.
func TestWeightsIdentityDigestsThroughTheMemo(t *testing.T) {
	path := filepath.Join(t.TempDir(), "weights.gguf")
	testcheck.NoError(t, os.WriteFile(path, []byte("weights fixture bytes"), 0o644))
	file, err := os.Open(path)
	testcheck.NoError(t, err)
	want, _, err := artifact.Identify(artifact.KindModel, file)
	file.Close()
	testcheck.NoError(t, err)
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

func TestWeightsIdentityReusesPersistedTensorDigest(t *testing.T) {
	path := filepath.Join(t.TempDir(), "weights.gguf")
	payload := []byte("registered tensor bytes")
	testcheck.NoError(t, os.WriteFile(path, payload, 0o600))
	store := overgodbtest.Open(t, t.TempDir())
	memo := NewMemo()
	identity := identifyLocation(path, artifact.KindTensorSet, map[string]fileIdentity{}, memo)
	if !identity.present {
		t.Fatal("tensor identity missing")
	}
	testcheck.NoError(t, PublishMemo(t.Context(), store, memo))
	loaded := LoadMemo(t.Context(), store)
	want, err := artifact.IdentifyBytes(artifact.KindModel, payload)
	testcheck.NoError(t, err)
	got, err := WeightsIdentity(path, loaded)
	if err != nil || got != want || loaded.dirty || len(loaded.entries) != 1 {
		t.Fatalf("persisted tensor digest was not reused: got=%s want=%s dirty=%v entries=%d error=%v", got, want, loaded.dirty, len(loaded.entries), err)
	}
	payload = append(payload, " changed"...)
	testcheck.NoError(t, os.WriteFile(path, payload, 0o600))
	want, err = artifact.IdentifyBytes(artifact.KindModel, payload)
	testcheck.NoError(t, err)
	got, err = WeightsIdentity(path, loaded)
	if err != nil || got != want || !loaded.dirty {
		t.Fatalf("changed bytes reused a stale digest: got=%s want=%s dirty=%v error=%v", got, want, loaded.dirty, err)
	}
}
