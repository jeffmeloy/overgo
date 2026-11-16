//go:build !windows

package processcontrol

import "errors"

// MachineCommit reports that the machine's commit is read on Windows.
func MachineCommit() (left, limit uint64, err error) {
	return 0, 0, errors.New("processcontrol: the machine's commit is read on Windows")
}
