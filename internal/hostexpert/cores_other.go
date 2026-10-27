//go:build !windows

package hostexpert

// physicalCores answers no cores off Windows: the pool's workers are not pinned there.
func physicalCores() []int { return nil }

// pinThread does nothing off Windows.
func pinThread(int) {}
