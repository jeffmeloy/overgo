// Package deviceprobe owns coarse physical-device diagnostics that are not
// available through an executor's allocation accounting.
package deviceprobe

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// Memory is the nvidia-smi memory state for the first visible GPU.
type Memory struct {
	UsedMiB int
	FreeMiB int
}

// MeasureMemory reads used and free memory with one nvidia-smi process. It
// returns errors rather than manufacturing measurements when the probe fails.
func MeasureMemory() (Memory, error) {
	out, err := exec.Command(
		"nvidia-smi",
		"--query-gpu=memory.used,memory.free",
		"--format=csv,noheader,nounits",
	).Output()
	if err != nil {
		return Memory{}, fmt.Errorf("nvidia-smi memory query: %w", err)
	}
	return parseMemory(out)
}

func parseMemory(out []byte) (Memory, error) {
	line, _, _ := strings.Cut(strings.TrimSpace(string(out)), "\n")
	fields := strings.Split(line, ",")
	if len(fields) != 2 {
		return Memory{}, fmt.Errorf("nvidia-smi memory query returned %q", line)
	}
	used, err := strconv.Atoi(strings.TrimSpace(fields[0]))
	if err != nil {
		return Memory{}, fmt.Errorf("parse nvidia-smi used memory %q: %w", fields[0], err)
	}
	free, err := strconv.Atoi(strings.TrimSpace(fields[1]))
	if err != nil {
		return Memory{}, fmt.Errorf("parse nvidia-smi free memory %q: %w", fields[1], err)
	}
	return Memory{UsedMiB: used, FreeMiB: free}, nil
}
