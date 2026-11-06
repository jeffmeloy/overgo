package discovery

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
)

// TestMemoReusesOnlyUnchangedFiles pins the memo's claim: a remembered digest
// is served only while size and modification time match the hashing stat, and
// any change invalidates it.
func TestMemoReusesOnlyUnchangedFiles(t *testing.T) {
	memo := NewMemo()
	path := filepath.Join(t.TempDir(), "weights.bin")
	testcheck.NoError(t, os.WriteFile(path, []byte("original bytes"), 0o644))
	info := testcheck.Must(os.Stat(path))(t)
	identity := fileIdentity{size: 14, present: true}
	memo.record("key", info, identity)
	remembered, ok := memo.lookup("key", info)
	if !ok || remembered != identity {
		t.Fatalf("unchanged file not reused: %v %v", remembered, ok)
	}
	testcheck.NoError(t, os.WriteFile(path, []byte("changed content!!"), 0o644))
	future := time.Now().Add(2 * time.Second)
	testcheck.NoError(t, os.Chtimes(path, future, future))
	changed := testcheck.Must(os.Stat(path))(t)
	if _, ok := memo.lookup("key", changed); ok {
		t.Fatal("changed file served a stale digest")
	}
	var absent *Memo
	if _, ok := absent.lookup("key", info); ok {
		t.Fatal("nil memo claimed a digest")
	}
	absent.record("key", info, identity)
}

// TestMemoClaimNamesTheRememberedIdentityWithoutReadingTheFile gives a claim
// for the file as hashed, and none for a changed, absent or unremembered one;
// the memo has no copy of the bytes to give it from.
func TestMemoClaimNamesTheRememberedIdentityWithoutReadingTheFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "weights.gguf")
	testcheck.NoError(t, os.WriteFile(path, []byte("original bytes"), 0o644))
	info := testcheck.Must(os.Stat(path))(t)
	id := testcheck.Must(artifact.IdentifyBytes(artifact.KindTensorSet, []byte("original bytes")))(t)
	memo := NewMemo()
	if _, _, ok := memo.Claim(path); ok {
		t.Fatal("an empty memo made a claim")
	}
	memo.record(path+"\x00"+artifact.KindTensorSet.String(), info, fileIdentity{id: id, size: 14, present: true})
	claimed, size, ok := memo.Claim(path)
	if !ok || claimed != id || size != 14 {
		t.Fatalf("claim = (%s, %d, %t), want (%s, 14, true)", claimed, size, ok, id)
	}
	if _, _, ok := memo.Claim(filepath.Dir(path)); ok {
		t.Fatal("a directory was claimed")
	}
	if _, _, ok := (*Memo)(nil).Claim(path); ok {
		t.Fatal("a nil memo made a claim")
	}
	future := time.Now().Add(2 * time.Second)
	testcheck.NoError(t, os.Chtimes(path, future, future))
	if _, _, ok := memo.Claim(path); ok {
		t.Fatal("a file with a new modification time kept its claim")
	}
	testcheck.NoError(t, os.Remove(path))
	if _, _, ok := memo.Claim(path); ok {
		t.Fatal("an absent file kept its claim")
	}
}
