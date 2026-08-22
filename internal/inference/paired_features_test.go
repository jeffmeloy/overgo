package inference

import (
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

func TestPairedFeatureLogitRowSkipsLastTokenRow(t *testing.T) {
	const (
		vocabulary = 3
		rows       = 3
		rowIndex   = 1
	)
	data := make([]float32, vocabulary*rows)
	for index := range data {
		data[index] = float32(index + 1)
	}
	logits := reference.Value{
		Shape: tensor.MustShape(vocabulary, rows),
		Data:  data,
	}
	row, err := pairedFeatureLogitRow(logits, rowIndex, vocabulary)
	if err != nil {
		t.Fatal(err)
	}
	wantStart := data[rowIndex*vocabulary]
	wantEnd := data[(rowIndex+1)*vocabulary-1]
	if len(row) != vocabulary || row[0] != wantStart || row[vocabulary-1] != wantEnd {
		t.Fatalf("unexpected paired-feature proposal row: %v", row)
	}
	if _, err := pairedFeatureLogitRow(logits, rows, vocabulary); err == nil {
		t.Fatal("accepted out-of-range paired-feature proposal row")
	}
}
