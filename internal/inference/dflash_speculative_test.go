package inference

import (
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

func TestDFlashLogitRowSkipsLastTokenRow(t *testing.T) {
	logits := reference.Value{
		Shape: tensor.MustShape(3, 3),
		Data:  []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
	}
	row, err := dflashLogitRow(logits, 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(row) != 3 || row[0] != 4 || row[2] != 6 {
		t.Fatalf("unexpected DFlash proposal row: %v", row)
	}
	if _, err := dflashLogitRow(logits, 3, 3); err == nil {
		t.Fatal("accepted out-of-range DFlash proposal row")
	}
}
