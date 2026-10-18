package modeldevice

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
	"overgo/internal/tensor/dtype"
)

// byteReads counts how often each byte of a file was read.
type byteReads struct {
	mu     sync.Mutex
	counts []int
}

func (b *byteReads) add(offset uint64, n int) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for index := offset; index < offset+uint64(n) && index < uint64(len(b.counts)); index++ {
		b.counts[index]++
	}
}

func (b *byteReads) snapshot() []int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return slices.Clone(b.counts)
}

// countedReader counts the reads of a file parsed from memory.
type countedReader struct {
	source io.ReaderAt
	reads  *byteReads
}

func (r countedReader) ReadAt(buffer []byte, offset int64) (int, error) {
	n, err := r.source.ReadAt(buffer, offset)
	r.reads.add(uint64(offset), n)
	return n, err
}

// countedFile counts the reads of a shard the pass opened by its path.
type countedFile struct {
	shardFile
	reads *byteReads
}

func (f countedFile) readAt(buffer []byte, offset uint64) (int, error) {
	n, err := f.shardFile.readAt(buffer, offset)
	f.reads.add(offset, n)
	return n, err
}

// TestExpertFillReadsEachRangeOnce follows the claimed open of a model with
// routed experts: resolution parses the directory and claims the file's
// identity without reading tensor bytes, and the open's one pass hashes the
// file, uploads the other weights and fills the experts' card and pinned host
// slots from the bytes it hashes. Every byte of every expert bank is read
// once across both, whether the pass reads the file from memory or opens the
// shard itself, and every slot holds the file's bytes. A byte changed under
// the claim, on the card's share or the host's, leaves no residency, no
// weight and no allocation.
func TestExpertFillReadsEachRangeOnce(t *testing.T) {
	random := rand.New(rand.NewPCG(23, 29))
	source := writeFixture(t, random, []fixtureTensor{
		// An unrouted table, as Flash-Next's n-gram embeddings are, holds the
		// experts past the 1 MiB that parsing the directory reads at once.
		{name: "table", shape: []uint64{512, 640}, kind: dtype.F32},
		{name: "attn", shape: []uint64{64, 2}, kind: dtype.Q8_0, raw: true},
		// A gate expert is smaller than its 256-byte slot; down and up experts fill theirs.
		{name: "gate", shape: []uint64{2, 2, 3}, kind: dtype.BF16},
		{name: "norm", shape: []uint64{16}, kind: dtype.F32, f32: true},
		{name: "down", shape: []uint64{16, 16, 3}, kind: dtype.F32},
		{name: "up", shape: []uint64{32, 8, 3}, kind: dtype.F32},
	}, 30, 5)
	banks := []string{"gate", "down", "up"}
	const (
		experts, pointerBytes    = 3, 8
		gateStride, matrixStride = 256, 1024
		// The card holds expert 0 of every bank and expert 1 of the gate bank,
		// so the pass splits each bank between the card and host memory.
		limit = 3*experts*pointerBytes + 2*gateStride + 2*matrixStride
		// Reads of 100 bytes cut experts, banks and the card's share apart.
		chunk = 100
	)
	card := map[string][]uint32{"gate": {0, 1}, "down": {0}, "up": {0}}
	claim := []artifact.ID{identity(t, source.data)}
	// pass resolves the file the way a claimed open does and runs the open's
	// pass over data, in memory or from a file the pass opens itself. It
	// returns the reads of each byte by resolution and by both together.
	pass := func(t *testing.T, fromFile bool, change func(data []byte)) (resolution, total []int, mem *fakeExpertDevice, s stores, err error) {
		t.Helper()
		data := slices.Clone(source.data)
		reads := &byteReads{counts: make([]int, len(data))}
		var file *gguf.File
		path := filepath.Join(t.TempDir(), "experts.gguf")
		if fromFile {
			if err := os.WriteFile(path, data, 0o600); err != nil {
				t.Fatal(err)
			}
			file, err = gguf.Open(path)
		} else {
			file, err = gguf.Parse(countedReader{bytes.NewReader(data), reads}, uint64(len(data)), gguf.DefaultOptions())
		}
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = file.Close() })
		resolution = reads.snapshot()
		if change != nil {
			change(data)
			if fromFile {
				if err := os.WriteFile(path, data, 0o600); err != nil {
					t.Fatal(err)
				}
			}
		}
		mem = newFakeExpertDevice()
		s = newStores(mem)
		s.raw = mem.weights()
		loader := NewLoader(file)
		loader.chunkBytes = chunk
		loader.open = func(path string) (shardFile, uint64, error) {
			opened, align, err := openShardFile(path)
			return countedFile{opened, reads}, align, err
		}
		source.route(loader, s, file)
		var infos []gguf.TensorInfo
		for _, name := range banks {
			info, _ := file.Tensor(name)
			infos = append(infos, info)
		}
		loader.Experts(s.raw, infos, limit)
		err = loader.Load(t.Context(), claim)
		return resolution, reads.snapshot(), mem, s, err
	}
	parsed := source.parse(t)
	bankRange := func(name string) (uint64, uint64) {
		info, _ := parsed.Tensor(name)
		start := parsed.Shards()[0].DataOffset + info.Offset
		return start, start + info.Size
	}
	for _, fromFile := range []bool{false, true} {
		t.Run(fmt.Sprintf("from-file=%t", fromFile), func(t *testing.T) {
			resolution, total, mem, s, err := pass(t, fromFile, nil)
			if err != nil {
				t.Fatal(err)
			}
			for _, name := range banks {
				start, end := bankRange(name)
				for position := start; position < end; position++ {
					if resolution[position] != 0 || total[position] != 1 {
						t.Fatalf("%s byte %d read %d times by resolution and %d in all, want once in all", name, position, resolution[position], total[position])
					}
				}
			}
			for position, count := range total {
				if count-resolution[position] != 1 {
					t.Fatalf("the pass read byte %d %d times; it hashes each byte once", position, count-resolution[position])
				}
			}
			cache := s.raw.experts
			if cache == nil {
				t.Fatal("the pass published no residency")
			}
			for _, name := range banks {
				bank := cache.banks[name]
				table, err := mem.at(bank.table, experts*pointerBytes)
				if err != nil {
					t.Fatal(err)
				}
				for index := range uint32(experts) {
					pointer := driver.DevicePtr(binary.LittleEndian.Uint64(table[index*pointerBytes:]))
					held, err := mem.at(pointer, bank.bytes)
					want := source.payloads[name][uint64(index)*bank.bytes:][:bank.bytes]
					if err != nil || !bytes.Equal(held, want) {
						t.Fatalf("%s expert %d does not hold its file bytes: %v", name, index, err)
					}
					if (bank.hosted[index] == nil) != slices.Contains(card[name], index) {
						t.Fatalf("%s expert %d on the card %t, want the card to hold %v", name, index, bank.hosted[index] == nil, card[name])
					}
				}
			}
			if routed := source.check(t, parsed, s, mem.contents); routed != 2 {
				t.Fatalf("the pass checked %d other weights, want 2", routed)
			}
			if err := errors.Join(cache.close(), s.raw.deviceTensorStore.Close(), s.f32.Close()); err != nil || mem.liveCount() != 0 {
				t.Fatalf("close = %v with %d allocations live", err, mem.liveCount())
			}
		})
		for _, side := range []string{"card", "host"} {
			t.Run(fmt.Sprintf("from-file=%t changed-%s-expert", fromFile, side), func(t *testing.T) {
				// Expert 0 of the down bank is on the card; expert 2 of the gate bank in host memory.
				name, index := "down", uint64(0)
				if side == "host" {
					name, index = "gate", 2
				}
				start, _ := bankRange(name)
				info, _ := parsed.Tensor(name)
				position := start + index*info.Size/experts + 1
				_, _, mem, s, err := pass(t, fromFile, func(data []byte) { data[position] ^= 0x20 })
				if !errors.Is(err, artifact.ErrContentMismatch) {
					t.Fatalf("pass over a changed %s expert = %v, want a content mismatch", side, err)
				}
				if s.raw.HasExpertResidency() || s.loaded() != 0 || mem.liveCount() != 0 {
					t.Fatalf("after the refusal: residency %t, %d weights stored, %d allocations live", s.raw.HasExpertResidency(), s.loaded(), mem.liveCount())
				}
			})
		}
	}
}
