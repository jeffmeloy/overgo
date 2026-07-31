package gguf

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestWriteSplitRoundTripsLogicalModel(t *testing.T) {
	var sourceBytes bytes.Buffer
	inputs := make([]TensorData, 5)
	for index := range inputs {
		inputs[index] = TensorData{
			Name:  fmt.Sprintf("weight.%d", index),
			Shape: []uint64{4},
			Type:  DTypeF32,
			Data:  bytes.NewReader(bytes.Repeat([]byte{byte(index + 1)}, 16)),
		}
	}
	if err := Write(
		&sourceBytes,
		[]Metadata{{
			Key: "general.name",
			Value: Value{
				Type: ValueTypeString,
				Data: "split writer",
			},
		}},
		inputs,
		WriteOptions{},
	); err != nil {
		t.Fatal(err)
	}
	source, err := Parse(
		bytes.NewReader(sourceBytes.Bytes()),
		uint64(sourceBytes.Len()),
		DefaultOptions(),
	)
	if err != nil {
		t.Fatal(err)
	}

	directory := t.TempDir()
	prefix := filepath.Join(directory, "part")
	err = source.WriteSplit(
		func(index, count uint16) (io.WriteCloser, error) {
			return os.Create(formatSplitPath(prefix, index, count))
		},
		SplitOptions{
			MaxTensors:           2,
			NoTensorsInFirstFile: true,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	merged, err := Open(formatSplitPath(prefix, 0, 4))
	if err != nil {
		t.Fatal(err)
	}
	defer merged.Close()
	if merged.SplitCount != 4 || len(merged.Tensors) != 5 {
		t.Fatalf("split/tensors = %d/%d, want 4/5", merged.SplitCount, len(merged.Tensors))
	}
	name, ok := merged.MetadataValue("general.name")
	if !ok || name.Data != "split writer" {
		t.Fatalf("first-shard metadata = %#v", name)
	}
	wantShards := []uint16{1, 1, 2, 2, 3}
	for index, tensor := range merged.Tensors {
		if tensor.Shard != wantShards[index] {
			t.Fatalf("tensor %d shard = %d, want %d", index, tensor.Shard, wantShards[index])
		}
		data := make([]byte, tensor.Size)
		if err := merged.ReadTensorData(tensor, data); err != nil {
			t.Fatal(err)
		}
		want := bytes.Repeat([]byte{byte(index + 1)}, len(data))
		if !bytes.Equal(data, want) {
			t.Fatalf("tensor %d data = %x, want %x", index, data, want)
		}
	}
}

func TestPlanSplitsHonorsAlignedPayloadTarget(t *testing.T) {
	file := &File{
		Alignment: 32,
		Tensors: []TensorInfo{
			{Name: "a", Size: 16},
			{Name: "b", Size: 16},
			{Name: "c", Size: 16},
		},
	}
	partitions, err := file.planSplits(SplitOptions{MaxTensors: 10, MaxBytes: 64})
	if err != nil {
		t.Fatal(err)
	}
	if len(partitions) != 2 || len(partitions[0]) != 2 || len(partitions[1]) != 1 {
		t.Fatalf("partition sizes = %v", []int{len(partitions[0]), len(partitions[1])})
	}
}
