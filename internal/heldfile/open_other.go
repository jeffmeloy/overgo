//go:build !windows

package heldfile

import "os"

// open opens path for reading. A Unix system holds no file against its writers, so the file can still change
// while it is held.
func open(path string) (*os.File, error) { return os.Open(path) }
