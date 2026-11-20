package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestOpsF32IsSplitByFamily holds kernels/cuda/ops_f32.cu to what its compiler flags need it for: its declarations and
// its includes, no kernel or device code of its own, the kernels lying in family headers, ops_f32_*.cuh, that it
// includes once each and no other source includes. The module's PTX, which the kernel manifest pins, is the same bytes
// the one file compiled to. How long a family header may be is the landing's rules stage's to hold (rules-family), in a
// commit attempt's first seconds.
func TestOpsF32IsSplitByFamily(t *testing.T) {
	t.Parallel()
	directory := filepath.Join("..", "..", "kernels", "cuda")
	module := string(testcheck.Must(os.ReadFile(filepath.Join(directory, "ops_f32.cu")))(t))
	if strings.Contains(module, "__global__") || strings.Contains(module, "{") {
		t.Error("ops_f32.cu holds code of its own; keep its kernels in its family headers")
	}
	included := map[string]int{}
	for line := range strings.SplitSeq(module, "\n") {
		if name, found := strings.CutPrefix(line, `#include "ops_f32_`); found {
			included["ops_f32_"+strings.TrimSuffix(strings.TrimSpace(name), `"`)]++
		}
	}
	families := testcheck.Must(filepath.Glob(filepath.Join(directory, "ops_f32_*.cuh")))(t)
	if len(families) == 0 {
		t.Fatal("ops_f32.cu has no family headers")
	}
	for _, path := range families {
		name := filepath.Base(path)
		if included[name] != 1 {
			t.Errorf("ops_f32.cu includes %s %d times, want once", name, included[name])
		}
	}
	for _, path := range testcheck.Must(filepath.Glob(filepath.Join(directory, "*.cu*")))(t) {
		if name := filepath.Base(path); name != "ops_f32.cu" && strings.Contains(string(testcheck.Must(os.ReadFile(path))(t)), `#include "ops_f32_`) {
			t.Errorf("%s includes an ops_f32 family header, which compiles only in ops_f32.cu", name)
		}
	}
}
