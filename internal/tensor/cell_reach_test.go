package tensor

import (
	"slices"
	"testing"
)

// TestABlockTokenReadsToItsBlocksEnd holds the step records a cell attention writes where tokens of a bidirectional
// block read past their own cells (judge-speed/cell-blocks): six tokens under a window of three, those at cells 2 to 4
// a block. Each token is appended at its own cell, a block token reads from its own window's first cell to its block's
// last, and every other token reads as before. A reach before a token's own cell, past the step's last cell, or of
// another length than the cells is refused.
func TestABlockTokenReadsToItsBlocksEnd(t *testing.T) {
	t.Parallel()
	attributes := CellAttentionAttributes{Cells: []uint32{0, 1, 2, 3, 4, 5}, Capacity: CellPageCells, Window: 3, Reach: []uint32{0, 1, 4, 4, 4, 5}}
	// Each record: the cell appended, the cells held after it, the first cell read and how many.
	want := []uint32{0, 1, 0, 1, 1, 2, 0, 2, 2, 3, 0, 5, 3, 4, 1, 4, 4, 5, 2, 3, 5, 6, 3, 3}
	if got := attributes.steps(); !slices.Equal(got, want) {
		t.Fatalf("the step records are %v, want %v", got, want)
	}
	plain := attributes
	plain.Reach = nil
	if got := plain.steps(); got[2*QSAStepWords+3] != 3 || got[4*QSAStepWords+3] != 3 {
		t.Fatalf("without a reach the block's tokens read %v, want each to its own cell", got)
	}
	for name, reach := range map[string][]uint32{
		"before its own cell":              {0, 1, 1, 4, 4, 5},
		"past the step's last cell":        {0, 1, 6, 6, 6, 5},
		"of another length than the cells": {0, 1, 4},
	} {
		refused := attributes
		refused.Reach = reach
		if err := validateCellAttentionReach(refused); err == nil {
			t.Errorf("a reach %s was taken", name)
		}
	}
	if err := validateCellAttentionReach(attributes); err != nil {
		t.Fatalf("the block's reach was refused: %v", err)
	}
}
