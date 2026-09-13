//go:build windows

package sensenovarecipe

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"runtime"
	"testing"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/jsonfile"
	"overgo/internal/processmeasure"
)

func TestSenseNovaVariableRequestResources(t *testing.T) {
	cudatest.Require(t)
	if cudatest.MeasurementProcess(t, 0) {
		return
	}
	var oracle generationLeadershipOracle
	if err := jsonfile.Decode(senseNovaGenerationGold, &oracle); err != nil {
		t.Fatal(err)
	}
	base := GenerationRequest{Prompt: oracle.Request.Prompt, Width: oracle.Request.Width, Height: oracle.Request.Height, Steps: oracle.Request.Steps, Seed: oracle.Request.Seed, CFGScale: oracle.Request.CFGScale, TimestepShift: oracle.Request.TimestepShift}
	changed := base
	changed.Width += base.Width
	changed.Prompt += " in the snow"
	changed.Seed++
	_, _, generator := newProductionGenerationFixture(t)
	var library *driver.Library
	if err := generator.worker.Do(t.Context(), func(state *device.State) error { library = state.Driver; return nil }); err != nil {
		t.Fatal(err)
	}
	generate := func(name string, owner *Generator, request GenerationRequest) string {
		t.Helper()
		var free, total uint64
		if err := owner.worker.Do(t.Context(), func(state *device.State) error {
			var err error
			free, total, err = state.Driver.MemInfo()
			return err
		}); err != nil {
			t.Fatal(err)
		}
		runtime.GC()
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		beforeCounters := library.ExecutionStats()
		started := time.Now()
		plan, err := owner.prepare(t.Context(), request)
		if err != nil {
			t.Fatal(err)
		}
		features, err := owner.integrate(t.Context(), plan)
		if err != nil {
			t.Fatal(err)
		}
		image, err := owner.decode(t.Context(), features)
		if err != nil {
			t.Fatal(err)
		}
		wall := time.Since(started)
		runtime.ReadMemStats(&after)
		peak, err := processmeasure.SelfPeakWorkingSet()
		if err != nil {
			t.Fatal(err)
		}
		digest := fmt.Sprintf("%x", sha256.Sum256(image.Data))
		if image.Width != request.Width || image.Height != request.Height || len(image.Data) == 0 {
			t.Fatal("request geometry changed during generation")
		}
		memory := library.MemoryStats()
		row, err := json.Marshal(map[string]any{
			"case": name, "request": request, "png_sha256": digest, "wall_ns": wall.Nanoseconds(),
			"allocated_bytes": after.TotalAlloc - before.TotalAlloc, "allocation_calls": after.Mallocs - before.Mallocs,
			"process_lifetime_peak_host_bytes": peak, "device_free_before": free, "device_total": total,
			"idle_device_bytes": memory.CurrentBytes, "library_peak_device_bytes": memory.PeakBytes,
			"before_counters": beforeCounters, "after_counters": library.ExecutionStats(),
			"scope": "One complete request after model load. Device counters and peak belong to this generator's library; host peak covers this process lifetime. No matched latency comparison.",
		})
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("MEDIA_CAPACITY %s", row)
		return digest
	}
	first := generate("native", generator, base)
	if first != senseNovaProductionPNG {
		t.Fatalf("native PNG changed: %s", first)
	}
	wider := generate("changed", generator, changed)
	idle := library.MemoryStats().CurrentBytes
	if generate("native-repeat", generator, base) != first || generate("changed-repeat", generator, changed) != wider {
		t.Fatal("changed request state contaminated a repeated output")
	}
	if got := library.MemoryStats().CurrentBytes; got != idle {
		t.Fatalf("repeated shape retained additional device memory: %d -> %d", idle, got)
	}
	if err := generator.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if got := library.MemoryStats().CurrentBytes; got != 0 {
		t.Fatalf("closed generator retains %d device bytes", got)
	}
	_, _, fresh := newProductionGenerationFixture(t)
	if err := fresh.worker.Do(t.Context(), func(state *device.State) error { library = state.Driver; return nil }); err != nil {
		t.Fatal(err)
	}
	if generate("changed-fresh", fresh, changed) != wider {
		t.Fatal("reused generator differs from a fresh generator for the changed request")
	}
	if err := fresh.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if got := library.MemoryStats().CurrentBytes; got != 0 {
		t.Fatalf("closed fresh generator retains %d device bytes", got)
	}
	t.Log("resource transitions: native and changed requests retain exact repeat/fresh outputs; repeated larger shape has stable idle bytes; both generators close to zero")
}
