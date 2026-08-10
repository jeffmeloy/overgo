package routedlm

import "fmt"

// BlockPositions: multi-axis row positions from a modality mask and an image
// token-grid width. Text rows (route 0) advance the semantic time index
// causally; each contiguous nonzero-route block shares one time index and
// rasterizes (H, W) row-major over the grid width. Port of the adaptive
// compileCausalPrefixInput time/height/width accumulation.
func BlockPositions(mask []int, tokenWidth int) ([]RowPosition, error) {
	if len(mask) == 0 || tokenWidth <= 0 {
		return nil, fmt.Errorf("routed lm block positions: rows=%d token width=%d", len(mask), tokenWidth)
	}
	positions := make([]RowPosition, len(mask))
	time := -1
	blockIndex := 0
	for row, route := range mask {
		if route == 0 {
			time++
			positions[row] = RowPosition{Time: time}
			blockIndex = 0
			continue
		}
		if row == 0 || mask[row-1] == 0 {
			time++
			blockIndex = 0
		}
		positions[row] = RowPosition{
			Branch: 1, Time: time,
			H: blockIndex / tokenWidth, W: blockIndex % tokenWidth,
		}
		blockIndex++
	}
	return positions, nil
}

// BlockCausalEnds: nondecreasing semantic time indexes into the final
// visible row for each query. Attention may see every earlier row plus later
// rows in the same explicit time block. Ported verbatim from adaptive
// extmodel blockCausalEnds.
func BlockCausalEnds(timeIndexes []int) ([]int32, error) {
	if len(timeIndexes) == 0 {
		return nil, fmt.Errorf("block causal indexes are empty")
	}
	ends := make([]int32, len(timeIndexes))
	for start := 0; start < len(timeIndexes); {
		if timeIndexes[start] < 0 {
			return nil, fmt.Errorf("invalid block causal index at row %d", start)
		}
		end := start + 1
		for end < len(timeIndexes) && timeIndexes[end] == timeIndexes[start] {
			end++
		}
		if end < len(timeIndexes) && timeIndexes[end] < timeIndexes[start] {
			return nil, fmt.Errorf("block causal indexes decrease at row %d", end)
		}
		last := int32(end - 1)
		for row := start; row < end; row++ {
			ends[row] = last
		}
		start = end
	}
	return ends, nil
}

// BlockCausalWindows: per-row [start, end) attention windows from row
// positions — start is always 0 (full-prefix visibility) and end extends
// through the row's shared-time block, the blockCausalEnds semantics in the
// same window shape SegmentWindows produces.
func BlockCausalWindows(positions []RowPosition) ([][2]int, error) {
	times := make([]int, len(positions))
	for row, pos := range positions {
		times[row] = pos.Time
	}
	ends, err := BlockCausalEnds(times)
	if err != nil {
		return nil, err
	}
	windows := make([][2]int, len(ends))
	for row, end := range ends {
		windows[row] = [2]int{0, int(end) + 1}
	}
	return windows, nil
}

// SegmentWindows: per-row [start, end) attention windows from visual
// segments (the segmentRange rule applied to every row).
func SegmentWindows(segments [][2]int, tokens int) [][2]int {
	windows := make([][2]int, tokens)
	for row := range windows {
		start, end := segmentRange(segments, row, tokens)
		windows[row] = [2]int{start, end}
	}
	return windows
}
