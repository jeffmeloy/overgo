package gate

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"

	"overgo/internal/cuda/driver"
	"overgo/internal/processcontrol"
)

func (g *gateContext) admitTestResources(ctx context.Context, packages, environment []string) (func() error, error) {
	noop := func() error { return nil }
	if runtime.GOOS != "windows" {
		return noop, nil
	}
	graph, err := g.inputGraph()
	if err != nil {
		return nil, err
	}
	devices, err := graph.devicePackages(packages)
	if err != nil {
		return nil, err
	}
	if len(devices) == 0 {
		return noop, nil
	}
	// Exclusive measurement children cannot run beneath a shared batch lease.
	for _, entry := range environment {
		key, value, _ := strings.Cut(entry, "=")
		if (key == "OVERGO_CUDA_TEST" || key == "OVERGO_PROBE_TEST") && value == "1" {
			return nil, fmt.Errorf("host test batch requires %s unset; run opted-in measurements through the device lane", key)
		}
	}
	library, err := driver.Open()
	if err != nil {
		return nil, err
	}
	defer library.Close()
	if err := library.Init(); err != nil {
		return nil, err
	}
	info, err := library.DeviceInfo(0)
	if err != nil {
		return nil, err
	}
	g.deviceResource = info.UUID
	return admitSharedTestResource(ctx, info.UUID)
}

// admitSharedTestResource takes the shared lease, waiting on an exclusive
// holder until it releases or exits; the caller's context ends the wait.
func admitSharedTestResource(ctx context.Context, name string) (func() error, error) {
	var release func() error
	began := time.Now()
	fmt.Fprintf(os.Stderr, "gate: resource=%s mode=shared state=waiting\n", name)
	err := processcontrol.AwaitResource(ctx, func() error {
		var err error
		release, err = processcontrol.ShareResource(name)
		return err
	})
	if err != nil {
		return nil, fmt.Errorf("host test resource admission: %w", err)
	}
	fmt.Fprintf(os.Stderr, "gate: resource=%s mode=shared state=admitted wait=%s\n", name, time.Since(began).Round(time.Millisecond))
	return func() error {
		err := release()
		if err == nil {
			fmt.Fprintf(os.Stderr, "gate: resource=%s mode=shared state=not_busy scope=test_batch\n", name)
		}
		return err
	}, nil
}
