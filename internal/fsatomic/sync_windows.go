//go:build windows

package fsatomic

// SyncDirectory is a no-op: Windows does not expose directory
// FlushFileBuffers through os.File.Sync, and the staged file itself
// is flushed before its atomic rename is published.
func SyncDirectory(string) error { return nil }
