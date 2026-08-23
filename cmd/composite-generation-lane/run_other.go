//go:build !windows

package main

import "overgo/internal/runrecord"

func run() error {
	return runrecord.LaneError(runrecord.LaneUnavailable, "composite generation CUDA lane requires Windows CUDA runtime")
}
