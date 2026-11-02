package strataspec

// NonfiniteBlocks is the slices of a logit row the device's guard checks one word each (kNonfiniteBlocks).
const NonfiniteBlocks = 16

// NonfiniteRow finds the first of rows whose flags name a NaN or an infinity, flags[row*NonfiniteBlocks+slice] being set.
// The guard reads bits, and a row's top-k maximum cannot stand in for it: a NaN never enters that list, so a row with one
// NaN has a finite maximum.
func NonfiniteRow(flags []int32, rows int) (row int, found bool) {
	for row := range rows {
		for _, flag := range flags[row*NonfiniteBlocks : (row+1)*NonfiniteBlocks] {
			if flag != 0 {
				return row, true
			}
		}
	}
	return 0, false
}
