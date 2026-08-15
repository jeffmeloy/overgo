//go:build !windows

package processmeasure

import (
	"errors"
	"os"
)

func peakWorkingSet(*os.Process) (uint64, error) {
	return 0, errors.New("process peak measurement is unavailable on this platform")
}
