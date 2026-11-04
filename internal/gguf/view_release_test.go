package gguf

import (
	"bytes"
	"encoding/binary"
	"io"
	"maps"
	"slices"
	"testing"
)

// TestAViewKeepsNoBuiltTensorsOnceItsLoadEnds reads a view as a load does, each built tensor once front to back in
// reads that end inside it: a tensor read part way stays built, so the rest of it is not built again, and once every
// tensor is read through the view keeps nothing it built. A read after the load builds the tensor again, and gives
// the same bytes.
func TestAViewKeepsNoBuiltTensorsOnceItsLoadEnds(t *testing.T) {
	t.Parallel()
	const elements = 1024
	opened := map[string]int{}
	tensor := func(name string, value float32) (ViewTensor, []byte) {
		data := make([]byte, 4*elements)
		for index := range elements {
			binary.LittleEndian.PutUint32(data[4*index:], uint32(index)+uint32(value))
		}
		return ViewTensor{Name: name, Shape: []uint64{elements}, Type: DTypeF32, Open: func() (io.Reader, error) {
			opened[name]++
			return bytes.NewReader(data), nil
		}}, data
	}
	first, firstData := tensor("first", 1)
	second, secondData := tensor("second", 7)
	file, err := View(nil, []ViewTensor{first, second}, WriteOptions{}, "")
	if err != nil {
		t.Fatal(err)
	}
	// Opening reads the directory with a read-ahead that may build the first tensors; the load is counted from here.
	source, before := file.source.(*viewSource), maps.Clone(opened)
	read := func(name string, from, size uint64) []byte {
		t.Helper()
		info, _ := file.Tensor(name)
		part := make([]byte, size)
		if err := file.ReadTensorRange(info, from, part); err != nil {
			t.Fatal(err)
		}
		return part
	}
	half := uint64(2 * elements)
	got := slices.Concat(read("first", 0, half), read("first", half, half))
	if !bytes.Equal(got, firstData) || len(source.built) != 0 || opened["first"] > before["first"]+1 {
		t.Fatalf("after first was read through in two halves: equal=%v, kept %d built, built it %d times",
			bytes.Equal(got, firstData), len(source.built), opened["first"]-before["first"])
	}
	read("second", 0, half)
	if len(source.built) != 1 {
		t.Fatalf("second, read half way, keeps %d built tensors, want itself", len(source.built))
	}
	read("second", half, half)
	built := opened["second"]
	if len(source.built) != 0 || built > before["second"]+1 {
		t.Fatalf("the load that read every tensor through left %d built tensors, and built second %d times", len(source.built), built-before["second"])
	}
	if again := read("second", 0, 2*half); !bytes.Equal(again, secondData) || opened["second"] != built+1 {
		t.Fatalf("a read after the load gave equal=%v, built second %d more times, want it built again", bytes.Equal(again, secondData), opened["second"]-built)
	}
}
