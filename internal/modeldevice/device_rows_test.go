package modeldevice

import (
	"bytes"
	"math/rand/v2"
	"testing"

	"overgo/internal/tensor/dtype"
)

// TestPlaceRowsGathersRowsInTheirOrder places a Q8_0 matrix's rows out of file
// order, repeated and across the reads' chunks, as a draft head's vocabulary
// names them: the tensor holds each row's stored bytes where the order names
// it, and a row past the matrix or a second placement under the name is
// refused with no allocation left behind.
func TestPlaceRowsGathersRowsInTheirOrder(t *testing.T) {
	const width, count = 64, 2*placeRowsChunk + 5
	random := rand.New(rand.NewPCG(5, 6))
	source := writeFixture(t, random, []fixtureTensor{{name: "head", shape: []uint64{width, count}, kind: dtype.Q8_0}}, 0, 0)
	file, device := source.parse(t), newFakeDevice()
	weights := newStores(device).raw
	head, _ := file.Tensor("head")
	rows := []uint32{count - 1, 3, placeRowsChunk, 3, placeRowsChunk - 1, 0}
	placed, err := weights.PlaceRows(t.Context(), file, head, "draft.head", rows)
	if err != nil {
		t.Fatal(err)
	}
	rowBytes := head.Size / count
	if placed.Type != dtype.Q8_0 || placed.Shape[0] != width || placed.Shape[1] != uint64(len(rows)) || placed.Size != uint64(len(rows))*rowBytes {
		t.Fatalf("placed %v %v of %d bytes", placed.Type, placed.Shape, placed.Size)
	}
	var want []byte
	for _, row := range rows {
		want = append(want, source.payloads["head"][uint64(row)*rowBytes:uint64(row+1)*rowBytes]...)
	}
	value, ok := weights.Lookup("draft.head")
	if !ok || !bytes.Equal(device.contents(t, value), want) {
		t.Fatal("the placed tensor does not hold the named rows in their order")
	}
	for _, refused := range []struct {
		name string
		rows []uint32
	}{{"past.head", []uint32{1, count}}, {"draft.head", []uint32{1}}} {
		if _, err := weights.PlaceRows(t.Context(), file, head, refused.name, refused.rows); err == nil {
			t.Fatalf("placing %v under %q was not refused", refused.rows, refused.name)
		}
	}
	if device.liveCount() != 1 {
		t.Fatalf("%d allocations live, want the one placed", device.liveCount())
	}
}
