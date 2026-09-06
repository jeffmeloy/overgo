package projector

import (
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

// Native Gemma4 sliding_window_mask_function uses strict distance bounds.
// Transformers source SHA256 ccab8e2dd80b71e9ca34e2c87291e17c40a27c755006e554da2ebf70d6616916.
// The old ten-token fixture never reached the first excluded past key at query 12.
func TestGemma4AudioAttentionChunkBoundary(t *testing.T) {
	spec := Gemma4AudioTowerSpec{ChunkSize: 12, ContextLeft: 13}
	for _, test := range []struct {
		name                     string
		start, query, key, right int
		allowed                  bool
	}{
		{"first self", 0, 0, 0, 0, true},
		{"first chunk history", 0, 11, 0, 0, true},
		{"second chunk excluded boundary", 12, 12, 0, 0, false},
		{"second chunk history", 12, 12, 1, 0, true},
		{"second chunk self", 12, 12, 12, 0, true},
		{"causal future", 12, 12, 13, 0, false},
		{"sliding excluded boundary", 12, 23, 11, 0, false},
		{"sliding history", 12, 23, 12, 0, true},
		{"padded query", 24, 25, 24, 0, false},
		{"padded key", 24, 24, 25, 2, false},
		{"future inside window", 12, 12, 13, 2, true},
		{"future excluded boundary", 12, 12, 14, 2, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			current := spec
			current.ContextRight = test.right
			graph := &projectorGraphRuntime{builder: tensor.NewBuilder(), hostFeeds: make(map[*tensor.Tensor]reference.Value)}
			node := gemma4AudioMask(graph, current, 25, test.start, test.start/current.ChunkSize, 0)
			width := current.ChunkSize + current.ContextLeft - 1 + current.ContextRight
			offset := test.key - test.start + current.ContextLeft - 1
			value := graph.hostFeeds[node].Data[(test.query-test.start)*width+offset]
			if (value == 0) != test.allowed || !test.allowed && value >= 0 {
				t.Fatalf("query=%d key=%d mask=%g allowed=%t", test.query, test.key, value, test.allowed)
			}
		})
	}
}
