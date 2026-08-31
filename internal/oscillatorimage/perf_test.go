package oscillatorimage

import (
	"sort"
	"testing"
	"time"
)

// TestGenerateWall: load wall plus median-of-5 warm per-image walls on the
// real artifact. Evidence probe for the performance leg, not an assertion —
// the at-or-below verdict lives in the plan measurement.
func TestGenerateWall(t *testing.T) {
	dir := artifactDir(t)
	loadStart := time.Now()
	m, err := Load(dir)
	if err != nil {
		t.Skipf("UNAVAILABLE: artifact absent at %s: %v", dir, err)
	}
	loadWall := time.Since(loadStart)

	if _, err := generatePlanar(m, Request{Class: 0, Seed: 1}); err != nil {
		t.Fatal(err)
	}
	// The model is tiny; per-generate wall is below Windows timer
	// granularity, so each sample times a 30-generate batch.
	const repeats, batch = 5, 30
	walls := make([]time.Duration, 0, repeats)
	for r := range repeats {
		start := time.Now()
		for i := range batch {
			if _, err := generatePlanar(m, Request{Class: i % m.Cfg.NClasses, Seed: int64(42 + r)}); err != nil {
				t.Fatal(err)
			}
		}
		walls = append(walls, time.Since(start))
	}
	sort.Slice(walls, func(i, j int) bool { return walls[i] < walls[j] })
	t.Logf("load=%v per-generate median=%v min=%v max=%v (batch %d, %d repeats)",
		loadWall, walls[repeats/2]/batch, walls[0]/batch, walls[repeats-1]/batch, batch, repeats)
}
