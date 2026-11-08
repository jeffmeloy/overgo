package media

import (
	"slices"
	"testing"
)

// TestConditioningCacheBounds holds the cache to its limit: the least recently used text leaves first, a lookup keeps
// a text, an output over the limit is not kept, a text is kept apart under each encoder, and a kept text never leaves
// and holds no part of the limit.
func TestConditioningCacheBounds(t *testing.T) {
	t.Parallel()
	cache := ConditioningCache{Limit: 6, Kept: []string{"negative"}}
	values := func(fill float32, count int) []float32 {
		out := make([]float32, count)
		for index := range out {
			out[index] = fill
		}
		return out
	}
	cache.Store("umt5", "negative", values(8, 7))
	cache.Store("umt5", "a", values(1, 2))
	cache.Store("umt5", "b", values(2, 2))
	cache.Store("umt5", "c", values(3, 2))
	if hidden, ok := cache.Lookup("umt5", "a"); !ok || !slices.Equal(hidden, values(1, 2)) {
		t.Fatal("kept text lost", hidden, ok)
	}
	cache.Store("umt5", "d", values(4, 2))
	if _, ok := cache.Lookup("umt5", "b"); ok {
		t.Fatal("least recently used text kept past the limit")
	}
	for _, text := range []string{"a", "c", "d", "negative"} {
		if _, ok := cache.Lookup("umt5", text); !ok {
			t.Fatal("recent or kept text evicted", text)
		}
	}
	if cache.held != 6 {
		t.Fatal("held values", cache.held)
	}
	cache.Store("umt5", "a", values(9, 2))
	if hidden, _ := cache.Lookup("umt5", "a"); !slices.Equal(hidden, values(1, 2)) || cache.held != 6 {
		t.Fatal("a known text replaced its output or counted twice", hidden, cache.held)
	}
	if _, ok := cache.Lookup("qwen", "a"); ok {
		t.Fatal("a text another encoder made was given for this one")
	}
	cache.Store("umt5", "huge", values(5, 7))
	if _, ok := cache.Lookup("umt5", "huge"); ok || cache.held != 6 {
		t.Fatal("an output over the limit was kept", cache.held)
	}
}
