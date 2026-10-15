//go:build !windows

package pytorchzip

import "os"

// openAdmission leaves byte reinspection required where the platform has no write exclusion.
func openAdmission(path string) (*os.File, bool, error) {
	file, err := os.Open(path)
	return file, false, err
}
