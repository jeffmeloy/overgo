package automationcheck

import (
	"context"
	"path/filepath"
	"strings"

	"overgo/internal/runrecord"
)

const deviceImpact Fact = "capability:device"

// DeviceCheck returns the exclusive-device verification adapter.
func DeviceCheck(root string, paths []string, command Command) Check {
	return Check{
		Descriptor: Descriptor{
			Name: "device", Phase: runrecord.PhaseTest, Triggers: []Fact{deviceImpact},
			Inapplicable: "no kernel or device implementation changed",
			Resources:    []Resource{{Name: "device", Exclusive: true}},
		},
		Run: func(context.Context, Invocation) (bool, string, error) {
			_, err := command(root, "go", "run", "./cmd/device-lane", "-paths", strings.Join(paths, ","))
			return false, "", err
		},
	}
}

// DeviceImpact derives whether shipped paths require device evidence.
func DeviceImpact(paths []string) []Fact {
	for _, path := range paths {
		path = filepath.ToSlash(path)
		if strings.HasPrefix(path, "kernels/") || strings.HasPrefix(path, "internal/cuda/") || strings.Contains(path, "_cuda_windows") {
			return []Fact{deviceImpact}
		}
	}
	return nil
}
