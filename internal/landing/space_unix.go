//go:build !windows

package landing

import (
	"fmt"
	"syscall"
)

// VolumeFree reads how many bytes this process may still write on the volume
// that holds directory.
func VolumeFree(directory string) (uint64, error) {
	var volume syscall.Statfs_t
	if err := syscall.Statfs(directory, &volume); err != nil {
		return 0, fmt.Errorf("free space of %s: %w", directory, err)
	}
	// The two fields' types differ by system, so each is converted.
	return uint64(volume.Bavail) * uint64(volume.Bsize), nil
}
