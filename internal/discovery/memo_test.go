package discovery

import (
	"os"
	"path/filepath"
	"testing"
	"time"
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
