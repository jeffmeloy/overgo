package diffusionimage

import (
	"runtime"
	"sort"
	"testing"
	"time"
)

// TestForwardWall: load wall plus median-of-5 warm full-forward walls on the
// real artifact (1x3x32x32 — the vendor-parity input). Evidence probe for
// the performance leg; the at-or-below verdict lives in the plan record.
func TestForwardWall(t *testing.T) {
	requireLongTest(t)
	dir := artifactDir(t)
	loadStart := time.Now()
	m, err := Load(dir)
	if err != nil {
		t.Skipf("UNAVAILABLE: artifact absent at %s: %v", dir, err)
	}
	loadWall := time.Since(loadStart)

	g := loadGolden[struct {
		B, C, H, W int
		X, Out     []float32
	}](t, "real_forward")
	if _, err := m.Forward(g.X, g.B, g.H, g.W); err != nil { // warm
		t.Fatal(err)
	}
	const repeats = 5
	walls := make([]time.Duration, 0, repeats)
	for r := 0; r < repeats; r++ {
		start := time.Now()
		if _, err := m.Forward(g.X, g.B, g.H, g.W); err != nil {
			t.Fatal(err)
		}
		walls = append(walls, time.Since(start))
	}
	sort.Slice(walls, func(i, j int) bool { return walls[i] < walls[j] })
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	watermark := float64(mem.HeapAlloc) / (1 << 20)
	runtime.GC()
	runtime.ReadMemStats(&mem)
	t.Logf("load=%v forward median=%v min=%v max=%v (%d warm repeats, 1x3x32x32); heap watermark %.1f MB, live after GC %.1f MB (HeapSys %.1f MB)",
		loadWall, walls[repeats/2], walls[0], walls[repeats-1], repeats,
		watermark, float64(mem.HeapAlloc)/(1<<20), float64(mem.HeapSys)/(1<<20))
	runtime.KeepAlive(m) // model must stay live through the GC'd heap read
}
