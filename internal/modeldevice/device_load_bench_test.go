package modeldevice

import (
	"context"
	"os"
	"strconv"
	"sync"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testskip"
)

// The host benchmark of the load structures runs on a real GGUF with no device:
//
//	OVERGO_LOAD_BENCH_GGUF=<model.gguf> [OVERGO_LOAD_BENCH_UPLOAD_GBPS=8.5] [OVERGO_LOAD_BENCH_ALLOC_US=0] \
//	    [OVERGO_LOAD_BENCH_REPEATS=1] go test ./internal/modeldevice -run '^TestLoaderStructureBenchmark$' -count=1 -v -timeout 30m
//
// It logs seconds per structure and asserts only that the single pass proves the
// identity the whole-file hash found. Run it twice: the first run reads the
// file as the cache holds it, the second warm.
const (
	benchGGUFEnvironment    = "OVERGO_LOAD_BENCH_GGUF"
	benchUploadEnvironment  = "OVERGO_LOAD_BENCH_UPLOAD_GBPS"
	benchAllocEnvironment   = "OVERGO_LOAD_BENCH_ALLOC_US"
	benchRepeatsEnvironment = "OVERGO_LOAD_BENCH_REPEATS"
	// benchSleepGrain is the least upload debt worth a sleep: below it the
	// timer's resolution would dominate the bandwidth it stands for.
	benchSleepGrain = time.Millisecond
)

// pacedDevice discards uploads, taking the time a device would: bytesPerSecond
// for a copy and allocLatency for an allocation. The caller's goroutine pays
// it, as a worker round trip does.
type pacedDevice struct {
	mu           sync.Mutex
	next         driver.DevicePtr
	bytesPerSec  float64
	allocLatency time.Duration
	debt         time.Duration
}

func (d *pacedDevice) alloc(_ context.Context, size uint64) (driver.DevicePtr, error) {
	time.Sleep(d.allocLatency)
	d.mu.Lock()
	defer d.mu.Unlock()
	pointer := d.next
	d.next += driver.DevicePtr(size) + 256
	return pointer, nil
}

func (d *pacedDevice) write(_ context.Context, _ driver.DevicePtr, data []byte) error {
	d.mu.Lock()
	d.debt += time.Duration(float64(len(data)) / d.bytesPerSec * float64(time.Second))
	owed := d.debt
	if owed >= benchSleepGrain {
		d.debt = 0
	}
	d.mu.Unlock()
	if owed >= benchSleepGrain {
		time.Sleep(owed)
	}
	return nil
}

func (d *pacedDevice) free(context.Context, []driver.DevicePtr) error { return nil }

// oldStyleLoad is the structure the loader had before the single pass: each
// store's tensors in turn, a tensor at a time, a tensor up to a chunk read and
// uploaded in turn, a larger one read ahead of its uploads through two buffers.
func oldStyleLoad(ctx context.Context, file *gguf.File, memory deviceMemory, raw, converted []gguf.TensorInfo) error {
	var buffers [2][]byte
	for _, info := range raw {
		pointer, err := memory.alloc(ctx, info.Size)
		if err != nil {
			return err
		}
		if info.Size <= model.WeightChunkSize {
			if cap(buffers[0]) < int(info.Size) {
				buffers[0] = make([]byte, int(info.Size))
			}
			data := buffers[0][:info.Size]
			if err := file.ReadTensorRange(info, 0, data); err != nil {
				return err
			}
			if err := memory.write(ctx, pointer, data); err != nil {
				return err
			}
			continue
		}
		free := make(chan []byte, len(buffers))
		ready := make(chan []byte, len(buffers))
		for index := range buffers {
			if cap(buffers[index]) < model.WeightChunkSize {
				buffers[index] = make([]byte, model.WeightChunkSize)
			}
			free <- buffers[index][:model.WeightChunkSize]
		}
		var readErr error
		go func() {
			defer close(ready)
			for offset := uint64(0); offset < info.Size; {
				buffer := <-free
				length := min(uint64(model.WeightChunkSize), info.Size-offset)
				if readErr = file.ReadTensorRange(info, offset, buffer[:length]); readErr != nil {
					return
				}
				ready <- buffer[:length]
				offset += length
			}
		}()
		offset := uint64(0)
		for data := range ready {
			if err := memory.write(ctx, pointer+driver.DevicePtr(offset), data); err != nil {
				return err
			}
			offset += uint64(len(data))
			free <- data[:model.WeightChunkSize]
		}
		if readErr != nil {
			return readErr
		}
	}
	for _, info := range converted {
		value, err := model.LoadHostTensor(ctx, file, info)
		if err != nil {
			return err
		}
		encoded := encodeF32(value.Data, 0)
		pointer, err := memory.alloc(ctx, uint64(len(encoded)))
		if err != nil {
			return err
		}
		if err := memory.write(ctx, pointer, encoded); err != nil {
			return err
		}
	}
	return nil
}

func benchRate(t *testing.T, name string, fallback float64) float64 {
	t.Helper()
	text := os.Getenv(name)
	if text == "" {
		return fallback
	}
	value, err := strconv.ParseFloat(text, 64)
	if err != nil || value < 0 {
		t.Fatalf("%s = %q is not a non-negative number", name, text)
	}
	return value
}

func TestLoaderStructureBenchmark(t *testing.T) {
	path := os.Getenv(benchGGUFEnvironment)
	if path == "" {
		testskip.NotApplicable(t, "set "+benchGGUFEnvironment+" to a GGUF path")
	}
	// Measured upload rate of a pageable synchronous copy to the device
	// (15.33 GB in 1.8 s summed, gemma-4-12B-it-fp8-native at 7031d1b37).
	const measuredUploadGBps = 8.5
	uploadGBps := benchRate(t, benchUploadEnvironment, measuredUploadGBps)
	allocLatency := time.Duration(benchRate(t, benchAllocEnvironment, 0) * float64(time.Microsecond))
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	var raw, converted []gguf.TensorInfo
	var rawBytes uint64
	for _, info := range file.Tensors {
		if info.Dimensions == 2 && (info.Type == dtype.F16 || info.Type == dtype.BF16 || info.Type == dtype.F8E4M3 || info.Type.IsQuantized()) {
			raw = append(raw, info)
			rawBytes += info.Size
		} else {
			converted = append(converted, info)
		}
	}
	var fileBytes uint64
	for _, shard := range file.Shards() {
		fileBytes += shard.Size
	}
	t.Logf("file=%s bytes=%d raw_tensors=%d raw_bytes=%d converted_tensors=%d upload_GBps=%.1f alloc_us=%.0f",
		path, fileBytes, len(raw), rawBytes, len(converted), uploadGBps, float64(allocLatency)/float64(time.Microsecond))
	repeats := int(benchRate(t, benchRepeatsEnvironment, 1))
	// phase runs a structure repeats times and returns the fastest: the run the
	// least other work on the machine disturbed.
	phase := func(name string, run func()) time.Duration {
		fastest := time.Duration(0)
		for range max(repeats, 1) {
			started := time.Now()
			run()
			elapsed := time.Since(started)
			t.Logf("structure=%-34s seconds=%8.3f GB/s=%.2f", name, elapsed.Seconds(), float64(fileBytes)/1e9/elapsed.Seconds())
			if fastest == 0 || elapsed < fastest {
				fastest = elapsed
			}
		}
		return fastest
	}
	paced := func() deviceMemory {
		return &pacedDevice{next: 0x10000, bytesPerSec: uploadGBps * 1e9, allocLatency: allocLatency}
	}
	routed := func(memory deviceMemory) *Loader {
		loader := NewLoader(file)
		rawStore := &DeviceWeights{deviceTensorStore: newDeviceTensorStore(nil, "device")}
		convertedStore := &DeviceConvertedWeights{
			deviceTensorStore: newDeviceTensorStore(nil, "F32 device"), storage: dtype.F32, bytesPerValue: 4, encode: encodeF32,
		}
		rawStore.mem, convertedStore.mem = memory, memory
		loader.Raw(rawStore, raw)
		loader.Converted(convertedStore, converted)
		return loader
	}

	var claim []artifact.ID
	hashed := phase("whole-file hash (artifact.Identify)", func() {
		claim = nil
		for _, source := range file.SourcePaths() {
			held, err := os.Open(source)
			if err != nil {
				t.Fatal(err)
			}
			id, _, err := artifact.Identify(artifact.KindTensorSet, held)
			_ = held.Close()
			if err != nil {
				t.Fatal(err)
			}
			claim = append(claim, id)
		}
	})
	uploaded := phase("old upload pass (no hash)", func() {
		if err := oldStyleLoad(t.Context(), file, paced(), raw, converted); err != nil {
			t.Fatal(err)
		}
	})
	t.Logf("structure=%-34s seconds=%8.3f", "old: hash first, then upload", (hashed + uploaded).Seconds())
	phase("new pass, upload only (no claim)", func() {
		if err := routed(paced()).Load(t.Context(), nil); err != nil {
			t.Fatal(err)
		}
	})
	phase("new pass, hash only (no routes)", func() {
		if err := NewLoader(file).Load(t.Context(), claim); err != nil {
			t.Fatal(err)
		}
	})
	single := phase("new pass, hash while uploading", func() {
		if err := routed(paced()).Load(t.Context(), claim); err != nil {
			t.Fatal(err)
		}
	})
	t.Logf("structure=%-34s saved=%.3f seconds against the old two passes", "single pass", (hashed + uploaded - single).Seconds())
}
