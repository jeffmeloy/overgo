package executor

import "testing"

// TestGroupedSparseAttentionGeometry pins the launches the device test
// exercises: Flash-Next's prompt takes its whole 12-head groups a block, a
// short step a head a block, a 576-key head a narrower tile, and a value row
// wider than a block's accumulators the per-channel kernel.
func TestGroupedSparseAttentionGeometry(t *testing.T) {
	const (
		flashNextWidth, flashNextHeads, flashNextKVHeads = 256, 24, 2
		mlaKeyWidth, mlaValueWidth, mlaHeads             = 576, 512, 16
		promptQueries, stepQueries, mlaQueries           = 130, 2, 40
		wideValueWidth                                   = 8192
	)
	for _, tc := range []struct {
		name                                                     string
		keyWidth, valueWidth, queryHeads, keyValueHeads, queries uint32
		want                                                     sparseAttentionGrouped
		grouped                                                  bool
	}{
		{"flash-next prompt", flashNextWidth, flashNextWidth, flashNextHeads, flashNextKVHeads, promptQueries, sparseAttentionGrouped{chunkHeads: 12, tile: 16, sharedBytes: 46048}, true},
		{"flash-next step", flashNextWidth, flashNextWidth, flashNextHeads, flashNextKVHeads, stepQueries, sparseAttentionGrouped{chunkHeads: 1, tile: 16, sharedBytes: 33992}, true},
		{"mla step", mlaKeyWidth, mlaValueWidth, mlaHeads, 1, mlaQueries, sparseAttentionGrouped{chunkHeads: 1, tile: 8, sharedBytes: 37224}, true},
		{"wide value", 4, wideValueWidth, 1, 1, stepQueries, sparseAttentionGrouped{}, false},
		{"ungrouped heads", flashNextWidth, flashNextWidth, flashNextHeads + 1, flashNextKVHeads, promptQueries, sparseAttentionGrouped{}, false},
	} {
		got, grouped := groupedSparseAttention(tc.keyWidth, tc.valueWidth, tc.queryHeads, tc.keyValueHeads, tc.queries)
		if grouped != tc.grouped || got != tc.want {
			t.Errorf("%s: geometry %+v grouped %t, want %+v %t", tc.name, got, grouped, tc.want, tc.grouped)
		}
		if grouped && uint64(got.sharedBytes) > sparseAttentionGroupedSharedLimit {
			t.Errorf("%s: %d shared bytes exceed the %d limit", tc.name, got.sharedBytes, sparseAttentionGroupedSharedLimit)
		}
	}
}
