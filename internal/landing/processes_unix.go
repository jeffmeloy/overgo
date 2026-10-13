//go:build !windows

package landing

import "errors"

// Processes lists every process that runs on this machine. Off Windows it
// reads none and says so, and a caller reads that as a machine it could not
// read, never as a quiet one.
func Processes() ([]Process, error) {
	return nil, errors.New("the process list is read on Windows only")
}
