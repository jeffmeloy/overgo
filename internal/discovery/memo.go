package discovery

import (
	"os"
	"sync"
	"time"
)

// Memo remembers file digests across Servable calls for interactive surfaces.
// A remembered identity is reused only while the file's size and modification
// time are unchanged since it was hashed, so the claim it carries is exact:
// this is the digest of these bytes as last read, and any change to the file
// re-hashes. The one-shot CLI keeps hashing every file per invocation; the
// memo exists because a workbench that re-hashes tens of gigabytes on every
// catalog view is not usable as a primary interface.
type Memo struct {
	mu      sync.Mutex
	entries map[string]memoEntry
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

// record remembers one hashed identity under the stat that held while hashing.
func (m *Memo) record(key string, info os.FileInfo, identity fileIdentity) {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries[key] = memoEntry{identity: identity, size: info.Size(), modified: info.ModTime()}
}
