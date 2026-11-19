package modelswap

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/testcheck"
)

// TestAModelServerStatesWhatItLoads holds the need a model server's launch states (served-need) to the bytes of the
// model's files: a split GGUF's every shard, read from its first, and nothing for a servable that is no GGUF, whose
// server keeps the commit free at its start as its bound. Bound at that alone, 77.0 GiB of this machine's 99.6 GiB at
// rest, Qwen3.8-Flash-Next's server was refused before it served (2026-11-19).
func TestAModelServerStatesWhatItLoads(t *testing.T) {
	t.Parallel()
	var whole bytes.Buffer
	tensors := make([]gguf.TensorData, 5)
	for index := range tensors {
		tensors[index] = gguf.TensorData{Name: fmt.Sprintf("weight.%d", index), Shape: []uint64{256}, Type: gguf.DTypeF32,
			Data: bytes.NewReader(bytes.Repeat([]byte{byte(index + 1)}, 1024))}
	}
	testcheck.NoError(t, gguf.Write(&whole, nil, tensors, gguf.WriteOptions{}))
	model := testcheck.Must(gguf.Parse(bytes.NewReader(whole.Bytes()), uint64(whole.Len()), gguf.DefaultOptions()))(t)
	directory := t.TempDir()
	shard := func(index, count uint16) string {
		return filepath.Join(directory, fmt.Sprintf("model-%05d-of-%05d.gguf", int(index)+1, count))
	}
	var shards []string
	testcheck.NoError(t, model.WriteSplit(func(index, count uint16) (io.WriteCloser, error) {
		shards = append(shards, shard(index, count))
		return os.Create(shard(index, count))
	}, gguf.SplitOptions{MaxTensors: 2, NoTensorsInFirstFile: true}))
	var want uint64
	for _, path := range shards {
		want += uint64(testcheck.Must(os.Stat(path))(t).Size())
	}
	if need := servableNeed(shards[0]); len(shards) < 2 || need != want {
		t.Errorf("a model split into %d files of %d bytes states a need of %d bytes; want every shard's", len(shards), want, need)
	}
	other := filepath.Join(directory, "model.safetensors")
	testcheck.NoError(t, os.WriteFile(other, []byte("not a gguf"), 0o600))
	if need := servableNeed(other); need != 0 {
		t.Errorf("a servable that is no GGUF states a need of %d bytes; want none", need)
	}
}
