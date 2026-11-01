package latentimage

import (
	"slices"
	"testing"
)

// TestConditioningCacheBounds holds the cache to its limit: the least recently
// used prompt leaves first, a lookup keeps a prompt, and an output over the
// limit is not kept.
func TestConditioningCacheBounds(t *testing.T) {
	t.Parallel()
	cache := conditioningCache{limit: 6}
	values := func(fill float32, count int) []float32 {
		out := make([]float32, count)
		for index := range out {
			out[index] = fill
		}
		return out
	}
	cache.store("a", values(1, 2))
	cache.store("b", values(2, 2))
	cache.store("c", values(3, 2))
	if hidden, ok := cache.lookup("a"); !ok || !slices.Equal(hidden, values(1, 2)) {
		t.Fatal("kept prompt lost", hidden, ok)
	}
	cache.store("d", values(4, 2))
	if _, ok := cache.lookup("b"); ok {
		t.Fatal("least recently used prompt kept past the limit")
	}
	for _, prompt := range []string{"a", "c", "d"} {
		if _, ok := cache.lookup(prompt); !ok {
			t.Fatal("recent prompt evicted", prompt)
		}
	}
	if cache.held != 6 {
		t.Fatal("held values", cache.held)
	}
	cache.store("a", values(9, 2))
	if hidden, _ := cache.lookup("a"); !slices.Equal(hidden, values(1, 2)) || cache.held != 6 {
		t.Fatal("a known prompt replaced its output or counted twice", hidden, cache.held)
	}
	cache.store("huge", values(5, 7))
	if _, ok := cache.lookup("huge"); ok || cache.held != 6 {
		t.Fatal("an output over the limit was kept", cache.held)
	}
}
