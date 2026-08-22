package representation

// AxisGridPositions returns three axis-position rows: a zero temporal axis
// and row/column positions for media tokens following a zero-position prefix.
func AxisGridPositions(prefixRows, gridHeight, gridWidth int) [3][]uint32 {
	rows := prefixRows + gridHeight*gridWidth
	var positions [3][]uint32
	for axis := range positions {
		positions[axis] = make([]uint32, rows)
	}
	for row := prefixRows; row < rows; row++ {
		gridIndex := row - prefixRows
		positions[1][row] = uint32(gridIndex / gridWidth)
		positions[2][row] = uint32(gridIndex % gridWidth)
	}
	return positions
}
