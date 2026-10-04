package discovery

import (
	"os"
	"sync"
	"time"

	"overgo/internal/artifact"
)

// Memo remembers file digests across Servable calls for interactive surfaces.
// A remembered identity is reused only while the file's size and modification
// time are unchanged since it was hashed, so the claim it carries is exact:
// this is the digest of these bytes as last read. Callers assume immutable
// artifact files; a size or modification-time change forces hashing again.
// Callers without a memo hash every file. Catalog and report consumers can
// reuse it to avoid reading unchanged model files on each invocation.
type Memo struct {
	mu      sync.Mutex
	entries map[string]memoEntry
	// loaded counts the entries seeded from persisted evidence; dirty
	// reports hashing added identities since, so publishing is a no-op
	// for a memo that answered purely from what it loaded.
	loaded int
	dirty  bool
}

type memoEntry struct {
	identity fileIdentity
	size     int64
	modified time.Time
}

// NewMemo returns an empty digest memo.
func NewMemo() *Memo { return &Memo{entries: map[string]memoEntry{}} }

// lookup returns the remembered identity if the file still matches the stat
// under which it was hashed.
func (m *Memo) lookup(key string, info os.FileInfo) (fileIdentity, bool) {
	if m == nil {
		return fileIdentity{}, false
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	entry, ok := m.entries[key]
	if !ok || entry.size != info.Size() || !entry.modified.Equal(info.ModTime()) {
		return fileIdentity{}, false
	}
	return entry.identity, true
}

// Claim is the identity remembered for the weights file at path, and the size
// it had, while the file's size and modification time still match what was
// hashed. It never reads the file: whoever reads the bytes next must prove the
// claim, and a claim they do not carry must fail that read.
func (m *Memo) Claim(path string) (artifact.ID, uint64, bool) {
	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return artifact.ID{}, 0, false
	}
	identity, found := m.lookup(path+"\x00"+artifact.KindTensorSet.String(), info)
	return identity.id, identity.size, found && identity.present
}

// record remembers one hashed identity under the stat that held while hashing.
func (m *Memo) record(key string, info os.FileInfo, identity fileIdentity) {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries[key] = memoEntry{identity: identity, size: info.Size(), modified: info.ModTime()}
	m.dirty = true
}
