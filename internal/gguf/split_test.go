package gguf

import (
	"bytes"
	"encoding/binary"
	"io"
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

// TestShardsCoverEveryPhysicalByte lists one shard per physical file, with the
// sizes and data offsets that locate every tensor, and reads header and
// padding bytes through them: the bytes a content identity covers.
func TestShardsCoverEveryPhysicalByte(t *testing.T) {
	directory := t.TempDir()
	prefix := filepath.Join(directory, "fixture")
	paths := []string{formatSplitPath(prefix, 0, 2), formatSplitPath(prefix, 1, 2)}
	writeSplitFixture(t, paths[0], 0, 2, 3, []splitFixtureTensor{{name: "first", fill: 0x11}, {name: "second", fill: 0x22}})
	writeSplitFixture(t, paths[1], 1, 2, 3, []splitFixtureTensor{{name: "third", fill: 0x33}})
	file, err := Open(paths[0])
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	shards := file.Shards()
	if len(shards) != len(paths) {
		t.Fatalf("shards = %d, want %d", len(shards), len(paths))
	}
	for index, path := range paths {
		onDisk, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if shards[index].Size != uint64(len(onDisk)) {
			t.Fatalf("shard %d size = %d, want %d", index, shards[index].Size, len(onDisk))
		}
		whole := make([]byte, len(onDisk))
		if _, err := io.ReadFull(io.NewSectionReader(shards[index].Source, 0, int64(len(whole))), whole); err != nil || !bytes.Equal(whole, onDisk) {
			t.Fatalf("shard %d read = %v, equal %t", index, err, bytes.Equal(whole, onDisk))
		}
		if shards[index].DataOffset%32 != 0 || shards[index].DataOffset >= shards[index].Size {
			t.Fatalf("shard %d data offset = %d of %d", index, shards[index].DataOffset, shards[index].Size)
		}
	}
	for _, tensor := range file.Tensors {
		shard := shards[tensor.Shard]
		viaTensor := make([]byte, tensor.Size)
		if err := file.ReadTensorData(tensor, viaTensor); err != nil {
			t.Fatal(err)
		}
		viaShard := make([]byte, tensor.Size)
		at := int64(shard.DataOffset + tensor.Offset)
		if _, err := io.ReadFull(io.NewSectionReader(shard.Source, at, int64(len(viaShard))), viaShard); err != nil || !bytes.Equal(viaShard, viaTensor) {
			t.Fatalf("tensor %q through its shard = %v, equal %t", tensor.Name, err, bytes.Equal(viaShard, viaTensor))
		}
	}
	if got := len((*File)(nil).Shards()); got != 0 {
		t.Fatalf("nil file lists %d shards", got)
	}
}
