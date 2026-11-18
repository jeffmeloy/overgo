//go:build !windows

package capabilityruntime

import "os"

// holdFile opens path for reading. A Unix system holds no file against its writers, so the file can still change
// while it is held.
func holdFile(path string) (*os.File, error) { return os.Open(path) }
