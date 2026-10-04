package discovery

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/artifact"
)

// TestMemoReusesOnlyUnchangedFiles pins the memo's claim: a remembered digest
// is served only while size and modification time match the hashing stat, and
// any change invalidates it.
func TestMemoReusesOnlyUnchangedFiles(t *testing.T) {
	memo := NewMemo()
	path := filepath.Join(t.TempDir(), "weights.bin")
	if err := os.WriteFile(path, []byte("original bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	identity := fileIdentity{size: 14, present: true}
	memo.record("key", info, identity)
	remembered, ok := memo.lookup("key", info)
	if !ok || remembered != identity {
		t.Fatalf("unchanged file not reused: %v %v", remembered, ok)
	}
	if err := os.WriteFile(path, []byte("changed content!!"), 0o644); err != nil {
		t.Fatal(err)
	}
	future := time.Now().Add(2 * time.Second)
	if err := os.Chtimes(path, future, future); err != nil {
		t.Fatal(err)
	}
	changed, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
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
	if err := os.WriteFile(path, []byte("original bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	id, err := artifact.IdentifyBytes(artifact.KindTensorSet, []byte("original bytes"))
	if err != nil {
		t.Fatal(err)
	}
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
	if err := os.Chtimes(path, future, future); err != nil {
		t.Fatal(err)
	}
	if _, _, ok := memo.Claim(path); ok {
		t.Fatal("a file with a new modification time kept its claim")
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if _, _, ok := memo.Claim(path); ok {
		t.Fatal("an absent file kept its claim")
	}
}
