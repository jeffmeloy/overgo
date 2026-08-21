//go:build windows

package objectstore

// Windows does not expose directory FlushFileBuffers through os.File.Sync.
// The staged object itself is flushed before its atomic link is published.
func syncDirectory(string) error { return nil }
