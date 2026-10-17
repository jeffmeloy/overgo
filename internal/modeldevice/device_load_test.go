package modeldevice

import (
	"bytes"
	"context"
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
	"overgo/internal/model"
	"overgo/internal/tensor/dtype"
)

// fakeDevice stands in for device memory: it keeps every live allocation's
// bytes, rejects a write that leaves its allocation and a free of what is not
// live, and can fail the nth allocation or write.
type fakeDevice struct {
	mu        sync.Mutex
	next      driver.DevicePtr
	live      map[driver.DevicePtr][]byte
	allocs    int
	writes    int
	failAlloc int // the allocation that fails with an out-of-memory error; 0 for none
	maxLive   int // physical allocation slots, independent of the logical byte limit; 0 is unlimited
	failWrite int // the write that fails; 0 for none
	onWrite   func()
}

func newFakeDevice() *fakeDevice {
	return &fakeDevice{next: 0x10000, live: make(map[driver.DevicePtr][]byte)}
}

func (d *fakeDevice) alloc(_ context.Context, size uint64) (driver.DevicePtr, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.allocs++
	if d.allocs == d.failAlloc || d.maxLive > 0 && len(d.live) >= d.maxLive {
		return 0, &driver.ResultError{Operation: "injected allocation", Code: 2}
	}
	pointer := d.next
	d.next += driver.DevicePtr(size) + 512
	d.live[pointer] = make([]byte, size)
	return pointer, nil
}

func (d *fakeDevice) write(_ context.Context, destination driver.DevicePtr, data []byte) error {
	d.mu.Lock()
	onWrite := d.onWrite
	d.writes++
	failed := d.writes == d.failWrite
	d.mu.Unlock()
	if onWrite != nil {
		onWrite()
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if failed {
		return errors.New("injected write failure")
	}
	for base, held := range d.live {
		if destination >= base && uint64(destination-base)+uint64(len(data)) <= uint64(len(held)) {
			copy(held[destination-base:], data)
			return nil
		}
	}
	return fmt.Errorf("write of %d bytes at %#x leaves every allocation", len(data), uint64(destination))
}

func (d *fakeDevice) free(_ context.Context, pointers []driver.DevicePtr) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	var errs []error
	for _, pointer := range pointers {
		if _, live := d.live[pointer]; !live {
			errs = append(errs, fmt.Errorf("free of %#x, which is not live", uint64(pointer)))
		}
		delete(d.live, pointer)
	}
	return errors.Join(errs...)
}

func (d *fakeDevice) liveCount() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return len(d.live)
}

func (d *fakeDevice) contents(t *testing.T, value DeviceTensor) []byte {
	t.Helper()
	d.mu.Lock()
	defer d.mu.Unlock()
	held, live := d.live[value.Pointer]
	if !live {
		t.Fatalf("tensor %q is not live", value.Info.Name)
	}
	return slices.Clone(held)
}

// stores are the three stores a resident load fills, on one fake device.
type stores struct {
	raw       *DeviceWeights
	f32, bf16 *DeviceConvertedWeights
}

func newStores(memory deviceMemory) stores {
	result := stores{
		raw: &DeviceWeights{deviceTensorStore: newDeviceTensorStore(nil, "device")},
		f32: &DeviceConvertedWeights{
			deviceTensorStore: newDeviceTensorStore(nil, "F32 device"),
			storage:           dtype.F32, bytesPerValue: 4, encode: encodeF32,
		},
		bf16: &DeviceConvertedWeights{
			deviceTensorStore: newDeviceTensorStore(nil, "BF16 device"),
			storage:           dtype.BF16, bytesPerValue: 2, encode: encodeBF16,
		},
	}
	result.raw.mem, result.f32.mem, result.bf16.mem = memory, memory, memory
	return result
}

func (s stores) loaded() int {
	count := 0
	for _, store := range []*deviceTensorStore{s.raw.deviceTensorStore, s.f32.deviceTensorStore, s.bf16.deviceTensorStore} {
		count += len(store.tensors)
	}
	return count
}

// fixtureTensor is one tensor of a fixture file and where its bytes go.
type fixtureTensor struct {
	name  string
	shape []uint64
	kind  dtype.Type
	raw   bool // upload as stored
	f32   bool // convert into the F32 store
	bf16  bool // convert into the BF16 store
}

// fixture is a serialized GGUF with the payload each tensor holds.
type fixture struct {
	data     []byte
	tensors  []fixtureTensor
	payloads map[string][]byte
}

func randomBytes(random *rand.Rand, size uint64) []byte {
	data := make([]byte, size)
	for index := range data {
		data[index] = byte(random.Uint32())
	}
	return data
}

// writeFixture serializes the tensors, in order, with random payloads, a note
// of the given length in the metadata, and trailing bytes after the data.
func writeFixture(t *testing.T, random *rand.Rand, tensors []fixtureTensor, noteBytes, trailing int) fixture {
	t.Helper()
	inputs := make([]gguf.TensorData, len(tensors))
	payloads := make(map[string][]byte, len(tensors))
	for index, spec := range tensors {
		info, err := gguf.NewTensorInfo(spec.name, spec.kind, spec.shape)
		if err != nil {
			t.Fatal(err)
		}
		payloads[spec.name] = randomBytes(random, info.Size)
		inputs[index] = gguf.TensorData{Name: spec.name, Shape: spec.shape, Type: spec.kind, Data: bytes.NewReader(payloads[spec.name])}
	}
	metadata := []gguf.Metadata{
		{Key: "general.name", Value: gguf.Value{Type: gguf.ValueTypeString, Data: "load fixture"}},
		{Key: "fixture.note", Value: gguf.Value{Type: gguf.ValueTypeString, Data: string(randomBytes(random, uint64(noteBytes)))}},
	}
	var buffer bytes.Buffer
	if err := gguf.Write(&buffer, metadata, inputs, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	buffer.Write(randomBytes(random, uint64(trailing)))
	return fixture{data: buffer.Bytes(), tensors: tensors, payloads: payloads}
}

func (f fixture) parse(t *testing.T) *gguf.File {
	t.Helper()
	file, err := gguf.Parse(bytes.NewReader(f.data), uint64(len(f.data)), gguf.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	return file
}

// identity is the content identity of bytes, the way a whole-file hash gives it.
func identity(t *testing.T, data []byte) artifact.ID {
	t.Helper()
	id, _, err := artifact.Identify(artifact.KindTensorSet, bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	return id
}

// route registers the fixture's routes on a loader over the stores, in file order.
func (f fixture) route(loader *Loader, s stores, file *gguf.File) {
	f.routeShuffled(loader, s, file, nil)
}

// routeShuffled registers them in an order drawn from random, when given: the
// pass lays the tensors out in file order whatever order they were routed in.
func (f fixture) routeShuffled(loader *Loader, s stores, file *gguf.File, random *rand.Rand) {
	var raw, f32, bf16 []gguf.TensorInfo
	for _, spec := range f.tensors {
		info, _ := file.Tensor(spec.name)
		if spec.raw {
			raw = append(raw, info)
		}
		if spec.f32 {
			f32 = append(f32, info)
		}
		if spec.bf16 {
			bf16 = append(bf16, info)
		}
	}
	if random != nil {
		for _, infos := range [][]gguf.TensorInfo{raw, f32, bf16} {
			random.Shuffle(len(infos), func(i, j int) { infos[i], infos[j] = infos[j], infos[i] })
		}
	}
	loader.Raw(s.raw, raw)
	loader.Converted(s.f32, f32)
	loader.Converted(s.bf16, bf16)
}

// check holds every routed tensor on the device as the per-store loads this
// pass replaced left it: raw bytes as stored, converted bytes as the store's
// encoder makes them from the tensor's dequantized values. read returns a
// tensor's bytes from the device. It returns how many tensors it found.
func (f fixture) check(t *testing.T, file *gguf.File, s stores, read func(*testing.T, DeviceTensor) []byte) int {
	t.Helper()
	routed := 0
	for _, spec := range f.tensors {
		info, _ := file.Tensor(spec.name)
		if spec.raw {
			routed++
			value, found := s.raw.Lookup(spec.name)
			if !found || !bytes.Equal(read(t, value), f.payloads[spec.name]) {
				t.Fatalf("raw tensor %q differs from its file bytes (found %t)", spec.name, found)
			}
		}
		decoded, err := model.DecodeHostTensor(info, f.payloads[spec.name])
		if err != nil {
			t.Fatal(err)
		}
		for _, converted := range []struct {
			wanted bool
			store  *DeviceConvertedWeights
		}{{spec.f32, s.f32}, {spec.bf16, s.bf16}} {
			if !converted.wanted {
				continue
			}
			routed++
			value, found := converted.store.Lookup(spec.name)
			size := int(uint64(len(decoded.Data)) * converted.store.bytesPerValue)
			want := converted.store.encode(decoded.Data, size)
			if !found || !bytes.Equal(read(t, value), want) || value.Shape != decoded.Shape || value.Size != uint64(size) {
				t.Fatalf("%s tensor %q differs from its converted values (found %t)", converted.store.kind, spec.name, found)
			}
		}
	}
	return routed
}

// checkFake holds the fake device to exactly the routed tensors: nothing else
// is live, and every store holds what it was routed.
func (f fixture) checkFake(t *testing.T, file *gguf.File, s stores, device *fakeDevice) {
	t.Helper()
	routed := f.check(t, file, s, device.contents)
	if device.liveCount() != routed || s.loaded() != routed {
		t.Fatalf("live allocations = %d, stored tensors = %d, want %d", device.liveCount(), s.loaded(), routed)
	}
}

// mixedFixture holds every route kind: raw, converted, both, neither.
func mixedFixture(t *testing.T, random *rand.Rand) fixture {
	return writeFixture(t, random, []fixtureTensor{
		{name: "a.f16", shape: []uint64{8, 4}, kind: dtype.F16, raw: true},
		{name: "b.f32", shape: []uint64{5}, kind: dtype.F32, f32: true},
		{name: "c.q8", shape: []uint64{64, 3}, kind: dtype.Q8_0, raw: true, bf16: true},
		{name: "d.skipped", shape: []uint64{16, 2}, kind: dtype.BF16},
		{name: "e.both", shape: []uint64{3, 3}, kind: dtype.F32, f32: true, bf16: true},
		{name: "f.f16", shape: []uint64{100}, kind: dtype.F16, raw: true},
		{name: "g.tail", shape: []uint64{7}, kind: dtype.F32},
	}, 41, 0)
}

func TestLoaderUploadsEachRoutedTensor(t *testing.T) {
	random := rand.New(rand.NewPCG(1, 2))
	source := mixedFixture(t, random)
	for _, chunk := range []uint64{1, 7, 33, 64, 4096, 1 << 20} {
		for _, claimed := range []bool{false, true} {
			t.Run(fmt.Sprintf("chunk=%d claimed=%t", chunk, claimed), func(t *testing.T) {
				file, device := source.parse(t), newFakeDevice()
				s := newStores(device)
				loader := NewLoader(file)
				loader.chunkBytes = chunk
				source.route(loader, s, file)
				var expect []artifact.ID
				if claimed {
					expect = []artifact.ID{identity(t, source.data)}
				}
				if err := loader.Load(t.Context(), expect); err != nil {
					t.Fatal(err)
				}
				source.checkFake(t, file, s, device)
			})
		}
	}
}

// TestLoaderDigestMatchesIdentifyForRandomLayouts holds the pass's digest to
// artifact.Identify's over random tensor layouts, routes, chunk sizes, header
// lengths and trailing bytes: a load verifies against Identify's identity or
// refuses.
func TestLoaderDigestMatchesIdentifyForRandomLayouts(t *testing.T) {
	kinds := []dtype.Type{dtype.F32, dtype.F16, dtype.BF16, dtype.Q8_0}
	for seed := range uint64(60) {
		random := rand.New(rand.NewPCG(seed, 7))
		tensors := make([]fixtureTensor, random.IntN(11))
		for index := range tensors {
			kind := kinds[random.IntN(len(kinds))]
			width := uint64(1 + random.IntN(40))
			if kind == dtype.Q8_0 {
				width = 32 * uint64(1+random.IntN(3))
			}
			shape := []uint64{width}
			if random.IntN(2) == 0 {
				shape = append(shape, uint64(1+random.IntN(5)))
			}
			tensors[index] = fixtureTensor{
				name: fmt.Sprintf("t%d", index), shape: shape, kind: kind,
				raw: random.IntN(2) == 0, f32: random.IntN(3) == 0, bf16: random.IntN(4) == 0,
			}
		}
		source := writeFixture(t, random, tensors, random.IntN(200), random.IntN(70))
		file, device := source.parse(t), newFakeDevice()
		s := newStores(device)
		loader := NewLoader(file)
		loader.chunkBytes = uint64(1 + random.IntN(300))
		source.routeShuffled(loader, s, file, random)
		if err := loader.Load(t.Context(), []artifact.ID{identity(t, source.data)}); err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		source.checkFake(t, file, s, device)
	}
}

// TestLoaderRefusesATamperedByteAnywhere changes one byte of the file after
// its identity was claimed, in each place the identity covers, and requires
// the load to refuse and leave nothing on the device.
func TestLoaderRefusesATamperedByteAnywhere(t *testing.T) {
	random := rand.New(rand.NewPCG(3, 4))
	source := writeFixture(t, random, []fixtureTensor{
		{name: "raw", shape: []uint64{40}, kind: dtype.F16, raw: true},
		{name: "converted", shape: []uint64{5}, kind: dtype.F32, f32: true},
		{name: "skipped", shape: []uint64{9}, kind: dtype.F32},
		{name: "last", shape: []uint64{6}, kind: dtype.F32, raw: true},
	}, 30, 9)
	file := source.parse(t)
	shard := file.Shards()[0]
	infoOf := func(name string) gguf.TensorInfo { info, _ := file.Tensor(name); return info }
	place := func(name string, within uint64) int { return int(shard.DataOffset + infoOf(name).Offset + within) }
	cases := map[string]int{
		"header":                  16,
		"metadata":                int(shard.DataOffset) - 40,
		"raw tensor":              place("raw", 3),
		"converted tensor":        place("converted", 4),
		"unrouted tensor":         place("skipped", 0),
		"alignment padding":       place("converted", infoOf("converted").Size),
		"padding after last data": place("last", infoOf("last").Size),
		"trailing bytes":          len(source.data) - 1,
	}
	if place("raw", infoOf("raw").Size) == place("converted", 0) {
		t.Fatal("the fixture has no padding between tensors")
	}
	claim := []artifact.ID{identity(t, source.data)}
	for name, position := range cases {
		t.Run(name, func(t *testing.T) {
			// The file is parsed intact and changes under the claim afterwards:
			// the directory is as claimed, and only the bytes the pass reads differ.
			altered := fixture{data: slices.Clone(source.data), tensors: source.tensors, payloads: source.payloads}
			parsed, device := altered.parse(t), newFakeDevice()
			altered.data[position] ^= 0x40
			s := newStores(device)
			loader := NewLoader(parsed)
			altered.route(loader, s, parsed)
			err := loader.Load(t.Context(), claim)
			if !errors.Is(err, artifact.ErrContentMismatch) {
				t.Fatalf("load of a file with a changed %s = %v, want a content mismatch", name, err)
			}
			if device.allocs == 0 {
				t.Fatal("the pass refused before uploading anything, so it proved nothing about release")
			}
			if device.liveCount() != 0 || s.loaded() != 0 {
				t.Fatalf("after the refusal %d allocations are live and %d tensors stored, want none", device.liveCount(), s.loaded())
			}
			// Without a claim nothing is verified here: the whole-file hash that
			// precedes such a load is what notices the change.
			bare := newStores(newFakeDevice())
			bareLoader := NewLoader(parsed)
			altered.route(bareLoader, bare, parsed)
			if err := bareLoader.Load(t.Context(), nil); err != nil {
				t.Fatalf("claimless load: %v", err)
			}
		})
	}
	t.Run("untouched", func(t *testing.T) {
		parsed, device := source.parse(t), newFakeDevice()
		s := newStores(device)
		loader := NewLoader(parsed)
		source.route(loader, s, parsed)
		if err := loader.Load(t.Context(), claim); err != nil {
			t.Fatal(err)
		}
		source.checkFake(t, parsed, s, device)
	})
}

// TestLoaderRefusesClaimsThatDoNotFit covers a claim for other content, one
// that names the wrong number of shards, and an invalid one.
func TestLoaderRefusesClaimsThatDoNotFit(t *testing.T) {
	random := rand.New(rand.NewPCG(5, 6))
	source := mixedFixture(t, random)
	other := mixedFixture(t, random)
	for name, claim := range map[string][]artifact.ID{
		"other content": {identity(t, other.data)},
		"two shards":    {identity(t, source.data), identity(t, source.data)},
		"invalid":       {{}},
	} {
		t.Run(name, func(t *testing.T) {
			file, device := source.parse(t), newFakeDevice()
			s := newStores(device)
			loader := NewLoader(file)
			source.route(loader, s, file)
			if err := loader.Load(t.Context(), claim); err == nil {
				t.Fatal("load accepted a claim the file does not carry")
			}
			if device.liveCount() != 0 || s.loaded() != 0 {
				t.Fatalf("%d allocations live, %d tensors stored after a refused claim", device.liveCount(), s.loaded())
			}
		})
	}
}

// readRecorder records the ranges read from a file.
type readRecorder struct {
	source io.ReaderAt
	mu     sync.Mutex
	ranges [][2]int64
}

func (r *readRecorder) ReadAt(buffer []byte, offset int64) (int, error) {
	r.mu.Lock()
	r.ranges = append(r.ranges, [2]int64{offset, offset + int64(len(buffer))})
	r.mu.Unlock()
	return r.source.ReadAt(buffer, offset)
}

// TestLoaderReadsEachByteOnce is the single pass itself: with a claim the
// reads tile the whole file exactly once, in order, so the bytes hashed are the
// bytes uploaded; without one they are exactly the routed tensors.
func TestLoaderReadsEachByteOnce(t *testing.T) {
	random := rand.New(rand.NewPCG(7, 8))
	source := mixedFixture(t, random)
	for _, claimed := range []bool{true, false} {
		recorder := &readRecorder{source: bytes.NewReader(source.data)}
		file, err := gguf.Parse(recorder, uint64(len(source.data)), gguf.DefaultOptions())
		if err != nil {
			t.Fatal(err)
		}
		recorder.ranges = nil
		device := newFakeDevice()
		s := newStores(device)
		loader := NewLoader(file)
		loader.chunkBytes = 50
		source.route(loader, s, file)
		var expect []artifact.ID
		if claimed {
			expect = []artifact.ID{identity(t, source.data)}
		}
		if err := loader.Load(t.Context(), expect); err != nil {
			t.Fatal(err)
		}
		var reads, wantRoutedBytes int64
		for _, spec := range source.tensors {
			if spec.raw || spec.f32 || spec.bf16 {
				info, _ := file.Tensor(spec.name)
				wantRoutedBytes += int64(info.Size)
			}
		}
		next := int64(0)
		for _, span := range recorder.ranges {
			reads += span[1] - span[0]
			if claimed && span[0] != next {
				t.Fatalf("read [%d,%d) after %d: the pass skipped or repeated bytes", span[0], span[1], next)
			}
			if !claimed && span[0] < next {
				t.Fatalf("read [%d,%d) before %d: tensors were not read in file order", span[0], span[1], next)
			}
			next = span[1]
		}
		if claimed && reads != int64(len(source.data)) {
			t.Fatalf("claimed load read %d bytes of %d", reads, len(source.data))
		}
		if !claimed && reads != wantRoutedBytes {
			t.Fatalf("claimless load read %d bytes, the routed tensors hold %d", reads, wantRoutedBytes)
		}
	}
}

// TestLoaderReleasesEverythingOnFailure fails an allocation, a write and the
// context mid-pass, with and without a claim. Load returns only after it has
// joined every stage, so a stage that outlived a failed pass would hang here.
func TestLoaderReleasesEverythingOnFailure(t *testing.T) {
	random := rand.New(rand.NewPCG(9, 10))
	source := mixedFixture(t, random)
	claim := []artifact.ID{identity(t, source.data)}
	for _, claimed := range []bool{false, true} {
		var expect []artifact.ID
		if claimed {
			expect = claim
		}
		for failure := 1; failure <= 7; failure++ {
			t.Run(fmt.Sprintf("allocation %d claimed=%t", failure, claimed), func(t *testing.T) {
				file, device := source.parse(t), newFakeDevice()
				device.failAlloc = failure
				s := newStores(device)
				loader := NewLoader(file)
				source.route(loader, s, file)
				err := loader.Load(t.Context(), expect)
				if !driver.IsOutOfMemory(err) {
					t.Fatalf("load = %v, want the injected out-of-memory error", err)
				}
				if device.liveCount() != 0 || s.loaded() != 0 {
					t.Fatalf("%d allocations live, %d tensors stored", device.liveCount(), s.loaded())
				}
			})
			t.Run(fmt.Sprintf("write %d claimed=%t", failure, claimed), func(t *testing.T) {
				file, device := source.parse(t), newFakeDevice()
				device.failWrite = failure
				s := newStores(device)
				loader := NewLoader(file)
				loader.chunkBytes = 64
				source.route(loader, s, file)
				if err := loader.Load(t.Context(), expect); err == nil {
					t.Fatal("load survived a failed write")
				}
				if device.liveCount() != 0 || s.loaded() != 0 {
					t.Fatalf("%d allocations live, %d tensors stored", device.liveCount(), s.loaded())
				}
			})
		}
		t.Run(fmt.Sprintf("cancelled claimed=%t", claimed), func(t *testing.T) {
			file, device := source.parse(t), newFakeDevice()
			ctx, cancel := context.WithCancel(t.Context())
			device.onWrite = cancel
			s := newStores(device)
			loader := NewLoader(file)
			loader.chunkBytes = 16
			source.route(loader, s, file)
			if err := loader.Load(ctx, expect); !errors.Is(err, context.Canceled) {
				t.Fatalf("load = %v, want cancellation", err)
			}
			if device.liveCount() != 0 || s.loaded() != 0 {
				t.Fatalf("%d allocations live, %d tensors stored", device.liveCount(), s.loaded())
			}
		})
	}
}

// TestLoaderRejectsRoutesThatCannotLoad covers a tensor routed twice to one
// store, a descriptor that is not the file's, a tensor the store already holds,
// and a store closed before the pass ends.
func TestLoaderRejectsRoutesThatCannotLoad(t *testing.T) {
	random := rand.New(rand.NewPCG(11, 12))
	source := mixedFixture(t, random)
	t.Run("routed twice", func(t *testing.T) {
		file, device := source.parse(t), newFakeDevice()
		s := newStores(device)
		info, _ := file.Tensor("a.f16")
		loader := NewLoader(file)
		loader.Raw(s.raw, []gguf.TensorInfo{info, info})
		if err := loader.Load(t.Context(), nil); err == nil || device.allocs != 0 {
			t.Fatalf("load = %v after %d allocations", err, device.allocs)
		}
	})
	t.Run("not the file's descriptor", func(t *testing.T) {
		file, device := source.parse(t), newFakeDevice()
		s := newStores(device)
		info, _ := file.Tensor("a.f16")
		info.Offset++
		loader := NewLoader(file)
		loader.Raw(s.raw, []gguf.TensorInfo{info})
		if err := loader.Load(t.Context(), nil); err == nil || device.allocs != 0 {
			t.Fatalf("load = %v after %d allocations", err, device.allocs)
		}
	})
	t.Run("already loaded", func(t *testing.T) {
		file, device := source.parse(t), newFakeDevice()
		s := newStores(device)
		first := NewLoader(file)
		source.route(first, s, file)
		if err := first.Load(t.Context(), nil); err != nil {
			t.Fatal(err)
		}
		held := device.liveCount()
		second := NewLoader(file)
		source.route(second, s, file)
		if err := second.Load(t.Context(), nil); err == nil {
			t.Fatal("a store accepted tensors it already holds")
		}
		if device.liveCount() != held {
			t.Fatalf("live allocations = %d after the refused second load, want the first load's %d", device.liveCount(), held)
		}
		if err := errors.Join(s.raw.Close(), s.f32.Close(), s.bf16.Close()); err != nil || device.liveCount() != 0 {
			t.Fatalf("close = %v, %d live", err, device.liveCount())
		}
	})
	t.Run("closed store", func(t *testing.T) {
		file, device := source.parse(t), newFakeDevice()
		s := newStores(device)
		if err := s.raw.Close(); err != nil {
			t.Fatal(err)
		}
		loader := NewLoader(file)
		source.route(loader, s, file)
		if err := loader.Load(t.Context(), nil); err == nil || device.liveCount() != 0 {
			t.Fatalf("load = %v with %d live", err, device.liveCount())
		}
	})
}

// TestLoaderVerifiesWithoutRoutes is the whole-file proof alone: the same pass,
// nothing to upload.
func TestLoaderVerifiesWithoutRoutes(t *testing.T) {
	random := rand.New(rand.NewPCG(13, 14))
	source := mixedFixture(t, random)
	claim := []artifact.ID{identity(t, source.data)}
	file := source.parse(t)
	if err := NewLoader(file).Load(t.Context(), claim); err != nil {
		t.Fatal(err)
	}
	source.data[len(source.data)/2] ^= 1
	if err := NewLoader(file).Load(t.Context(), claim); !errors.Is(err, artifact.ErrContentMismatch) {
		t.Fatalf("load of changed bytes = %v", err)
	}
}

// TestLoaderVerifiesEverySplitShard splits a model into three files, the first
// holding no tensors, and changes one byte in each file in turn.
func TestLoaderVerifiesEverySplitShard(t *testing.T) {
	random := rand.New(rand.NewPCG(15, 16))
	source := writeFixture(t, random, []fixtureTensor{
		{name: "a", shape: []uint64{8, 4}, kind: dtype.F16, raw: true},
		{name: "b", shape: []uint64{5}, kind: dtype.F32, f32: true},
		{name: "c", shape: []uint64{64, 3}, kind: dtype.Q8_0, raw: true},
		{name: "d", shape: []uint64{9}, kind: dtype.F32},
	}, 20, 0)
	whole := source.parse(t)
	directory := t.TempDir()
	prefix := filepath.Join(directory, "split")
	name := func(index, count uint16) string { return fmt.Sprintf("%s-%05d-of-%05d.gguf", prefix, index+1, count) }
	err := whole.WriteSplit(func(index, count uint16) (io.WriteCloser, error) {
		return os.Create(name(index, count))
	}, gguf.SplitOptions{MaxTensors: 2, NoTensorsInFirstFile: true})
	if err != nil {
		t.Fatal(err)
	}
	const shards = 3
	paths := make([]string, shards)
	var claim []artifact.ID
	for index := range paths {
		paths[index] = name(uint16(index), shards)
		onDisk, err := os.ReadFile(paths[index])
		if err != nil {
			t.Fatal(err)
		}
		claim = append(claim, identity(t, onDisk))
	}
	load := func(t *testing.T, expect []artifact.ID) (error, *fakeDevice, stores) {
		file, err := gguf.Open(paths[0])
		if err != nil {
			t.Fatal(err)
		}
		defer file.Close()
		device := newFakeDevice()
		s := newStores(device)
		loader := NewLoader(file)
		source.route(loader, s, file)
		loader.chunkBytes = 40
		err = loader.Load(t.Context(), expect)
		if err == nil {
			source.checkFake(t, file, s, device)
		}
		return err, device, s
	}
	t.Run("intact", func(t *testing.T) {
		if err, _, _ := load(t, claim); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("one claim for three shards", func(t *testing.T) {
		if err, _, _ := load(t, claim[:1]); err == nil {
			t.Fatal("load accepted one claim for three shards")
		}
	})
	for index := range paths {
		t.Run(fmt.Sprintf("shard %d changed", index), func(t *testing.T) {
			onDisk, err := os.ReadFile(paths[index])
			if err != nil {
				t.Fatal(err)
			}
			changed := slices.Clone(onDisk)
			position := len(changed) - 1
			if index == 0 {
				// The first shard holds only the header: change its model name.
				position = bytes.Index(changed, []byte("load fixture"))
			}
			changed[position] ^= 0x10
			if err := os.WriteFile(paths[index], changed, 0o600); err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = os.WriteFile(paths[index], onDisk, 0o600) })
			err, device, s := load(t, claim)
			if !errors.Is(err, artifact.ErrContentMismatch) {
				t.Fatalf("load = %v, want a content mismatch", err)
			}
			if device.liveCount() != 0 || s.loaded() != 0 {
				t.Fatalf("%d allocations live, %d tensors stored after the refusal", device.liveCount(), s.loaded())
			}
		})
	}
}
