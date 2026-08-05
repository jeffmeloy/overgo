package gguf

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type splitFixtureTensor struct {
	name string
	fill byte
}

func TestOpenLoadsAndReadsSplitGGUF(t *testing.T) {
	directory := t.TempDir()
	prefix := filepath.Join(directory, "fixture")
	firstPath := formatSplitPath(prefix, 0, 2)
	secondPath := formatSplitPath(prefix, 1, 2)
	writeSplitFixture(t, firstPath, 0, 2, 3, []splitFixtureTensor{
		{name: "first", fill: 0x11},
		{name: "second", fill: 0x22},
	})
	writeSplitFixture(t, secondPath, 1, 2, 3, []splitFixtureTensor{
		{name: "third", fill: 0x33},
	})

	file, err := Open(firstPath)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	paths := file.SourcePaths()
	if len(paths) != 2 || paths[0] != firstPath || paths[1] != secondPath {
		t.Fatalf("source paths = %v", paths)
	}
	if file.SplitCount != 2 || len(file.Tensors) != 3 || file.DataSize != 96 {
		t.Fatalf(
			"split/count/data = %d/%d/%d",
			file.SplitCount,
			len(file.Tensors),
			file.DataSize,
		)
	}
	for index, expected := range []struct {
		name  string
		shard uint16
		fill  byte
	}{
		{name: "first", shard: 0, fill: 0x11},
		{name: "second", shard: 0, fill: 0x22},
		{name: "third", shard: 1, fill: 0x33},
	} {
		tensor := file.Tensors[index]
		if tensor.Name != expected.name || tensor.Shard != expected.shard {
			t.Fatalf("tensor %d = %+v", index, tensor)
		}
		data := make([]byte, tensor.Size)
		if err := file.ReadTensorData(tensor, data); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(data, bytes.Repeat([]byte{expected.fill}, len(data))) {
			t.Fatalf("tensor %q data = %x", tensor.Name, data)
		}
	}
	if _, err := Open(secondPath); err == nil ||
		!strings.Contains(err.Error(), "first split") {
		t.Fatalf("opening second split error = %v", err)
	}
}

func TestOpenRejectsMissingCorruptAndOversizedSplits(t *testing.T) {
	t.Run("missing part", func(t *testing.T) {
		prefix := filepath.Join(t.TempDir(), "missing")
		firstPath := formatSplitPath(prefix, 0, 2)
		writeSplitFixture(t, firstPath, 0, 2, 1, []splitFixtureTensor{
			{name: "first", fill: 1},
		})
		if _, err := Open(firstPath); err == nil ||
			!strings.Contains(err.Error(), "open GGUF split") {
			t.Fatalf("error = %v", err)
		}
	})
	t.Run("duplicate tensor", func(t *testing.T) {
		prefix := filepath.Join(t.TempDir(), "duplicate")
		firstPath := formatSplitPath(prefix, 0, 2)
		writeSplitFixture(t, firstPath, 0, 2, 2, []splitFixtureTensor{
			{name: "same", fill: 1},
		})
		writeSplitFixture(t, formatSplitPath(prefix, 1, 2), 1, 2, 2, []splitFixtureTensor{
			{name: "same", fill: 2},
		})
		if _, err := Open(firstPath); err == nil ||
			!strings.Contains(err.Error(), "duplicate tensor") {
			t.Fatalf("error = %v", err)
		}
	})
	t.Run("wrong total", func(t *testing.T) {
		prefix := filepath.Join(t.TempDir(), "total")
		firstPath := formatSplitPath(prefix, 0, 2)
		writeSplitFixture(t, firstPath, 0, 2, 3, []splitFixtureTensor{
			{name: "first", fill: 1},
		})
		writeSplitFixture(t, formatSplitPath(prefix, 1, 2), 1, 2, 3, []splitFixtureTensor{
			{name: "second", fill: 2},
		})
		if _, err := Open(firstPath); err == nil ||
			!strings.Contains(err.Error(), "declares 3 tensors but 2 were loaded") {
			t.Fatalf("error = %v", err)
		}
	})
	t.Run("configured limit", func(t *testing.T) {
		prefix := filepath.Join(t.TempDir(), "limited")
		firstPath := formatSplitPath(prefix, 0, 2)
		writeSplitFixture(t, firstPath, 0, 2, 1, []splitFixtureTensor{
			{name: "first", fill: 1},
		})
		options := DefaultOptions()
		options.MaxSplitFiles = 1
		if _, err := OpenWithOptions(firstPath, options); err == nil ||
			!strings.Contains(err.Error(), "split count 2 exceeds limit 1") {
			t.Fatalf("error = %v", err)
		}
	})
}

func TestSplitPathHelpersMatchPinnedNaming(t *testing.T) {
	path := `C:\models\name-00002-of-00004.gguf`
	prefix, err := splitPathPrefix(path, 1, 4)
	if err != nil {
		t.Fatal(err)
	}
	if prefix != `C:\models\name` ||
		formatSplitPath(prefix, 3, 4) != `C:\models\name-00004-of-00004.gguf` {
		t.Fatalf("prefix/path = %q/%q", prefix, formatSplitPath(prefix, 3, 4))
	}
	if _, err := splitPathPrefix("name.gguf", 0, 2); err == nil {
		t.Fatal("accepted non-split filename")
	}
}

func writeSplitFixture(
	t *testing.T,
	path string,
	index, count uint16,
	total uint32,
	tensors []splitFixtureTensor,
) {
	t.Helper()
	var buffer bytes.Buffer
	write := func(value any) {
		if err := binary.Write(&buffer, binary.LittleEndian, value); err != nil {
			t.Fatalf("write fixture: %v", err)
		}
	}
	writeString := func(value string) {
		write(uint64(len(value)))
		_, _ = buffer.WriteString(value)
	}
	writeMetadata := func(key string, valueType ValueType, value any) {
		writeString(key)
		write(uint32(valueType))
		write(value)
	}

	_, _ = buffer.WriteString(Magic)
	write(uint32(CurrentVersion))
	write(uint64(len(tensors)))
	metadataCount := uint64(3)
	if index == 0 {
		metadataCount = 5
	}
	write(metadataCount)
	if index == 0 {
		writeMetadata("general.alignment", ValueTypeUint32, uint32(32))
		writeString("general.name")
		write(uint32(ValueTypeString))
		writeString("split fixture")
	}
	writeMetadata("split.no", ValueTypeUint16, index)
	writeMetadata("split.count", ValueTypeUint16, count)
	writeMetadata("split.tensors.count", ValueTypeInt32, int32(total))

	for tensorIndex, tensor := range tensors {
		writeString(tensor.name)
		write(uint32(1))
		write(uint64(4))
		write(uint32(DTypeF32))
		write(uint64(tensorIndex * 32))
	}
	for buffer.Len()%32 != 0 {
		_ = buffer.WriteByte(0)
	}
	for _, tensor := range tensors {
		_, _ = buffer.Write(bytes.Repeat([]byte{tensor.fill}, 16))
		_, _ = buffer.Write(make([]byte, 16))
	}
	if err := os.WriteFile(path, buffer.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
}
