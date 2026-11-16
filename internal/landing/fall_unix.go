//go:build !windows

package landing

// watchChanges reports no change on this system, so a fall is read where the watch starts and stops.
func watchChanges(string, func()) (func() error, error) {
	return func() error { return nil }, nil
}
