package modeldevice

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"slices"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
	"overgo/internal/model"
)

const (
	// loadBufferCount is the chunk buffers shared by the read, hash and upload
	// stages: one per stage, two queued between each pair of stages, and the
	// slack that lets a slow allocation pass without stalling the hash.
	loadBufferCount = 6
	// loadQueueDepth is how many chunks wait between two stages.
	loadQueueDepth = 2
)

// Loader reads a model file once, in file order, and uploads each routed
// tensor to the stores it is routed to. Given the identity every shard must
// carry, the same pass feeds every byte of every shard, header, padding and
// unrouted tensors included, to that shard's hasher before the byte is
// uploaded or converted; it publishes nothing, and frees every allocation,
// unless every digest matches. A Loader runs once.
type Loader struct {
	file       *gguf.File
	chunkBytes uint64
	routed     []*segment
	byName     map[string]*segment
	err        error
}

// segment is a run of one shard's bytes: a routed tensor, or bytes that are
// only hashed (header, padding, unrouted tensors, trailing bytes).
type segment struct {
	shard  int
	offset uint64
	length uint64

	// A routed tensor's own fields.
	info     gguf.TensorInfo
	routes   []route
	pointers []driver.DevicePtr // raw routes' allocations, made at the first chunk
	storage  []byte             // the whole tensor, read for routes that convert
}

// route is one destination of a tensor's bytes: a raw device copy, or a
// converted one.
type route struct {
	store         *deviceTensorStore
	encode        convertedTensorEncoder // nil: the bytes upload as stored
	bytesPerValue uint64
}

// NewLoader starts a pass over the file with no tensors routed.
func NewLoader(file *gguf.File) *Loader {
	return &Loader{file: file, chunkBytes: model.WeightChunkSize, byName: make(map[string]*segment)}
}

// Raw routes tensors into the store as the file stores them.
func (l *Loader) Raw(store *DeviceWeights, infos []gguf.TensorInfo) {
	for _, info := range infos {
		l.route(info, route{store: store.deviceTensorStore})
	}
}

// Converted routes tensors into the store dequantized and encoded to its
// storage type. One tensor may take several routes and is read once.
func (l *Loader) Converted(store *DeviceConvertedWeights, infos []gguf.TensorInfo) {
	for _, info := range infos {
		l.route(info, route{store: store.deviceTensorStore, encode: store.encode, bytesPerValue: store.bytesPerValue})
	}
}

func (l *Loader) route(info gguf.TensorInfo, destination route) {
	known := l.byName[info.Name]
	if known == nil {
		known = &segment{info: info}
		l.byName[info.Name] = known
		l.routed = append(l.routed, known)
	}
	if slices.ContainsFunc(known.routes, func(other route) bool { return other.store == destination.store }) {
		l.err = fmt.Errorf("%s tensor %q is routed twice", destination.store.kind, info.Name)
		return
	}
	known.routes = append(known.routes, destination)
}

// converts reports whether a route needs the whole tensor decoded.
func (s *segment) converts() bool {
	return slices.ContainsFunc(s.routes, func(r route) bool { return r.encode != nil })
}

// Load runs the pass. expect is the identity of each shard in file order, or
// empty to load without verifying; a shard that carries another identity
// fails the load with artifact.ErrContentMismatch.
func (l *Loader) Load(ctx context.Context, expect []artifact.ID) error {
	if l.err != nil {
		return l.err
	}
	shards := l.file.Shards()
	if len(expect) != 0 && len(expect) != len(shards) {
		return fmt.Errorf("model load: %d claimed identities for %d shards", len(expect), len(shards))
	}
	run := &loadRun{shards: shards}
	var err error
	if run.segments, err = l.segments(shards, len(expect) != 0); err != nil {
		return err
	}
	for _, id := range expect {
		hasher, err := artifact.NewHasher(id.Kind())
		if err != nil {
			return err
		}
		run.hashers = append(run.hashers, hasher)
	}
	err = run.pipeline(ctx, l.chunkBytes)
	if err == nil {
		err = run.verify(expect)
	}
	if err == nil {
		err = run.publish()
	}
	if err != nil {
		return errors.Join(err, run.release())
	}
	return nil
}

// segments partitions the shards in file order: the routed tensors and, when
// hashing, the bytes between them. The directory the file was parsed from
// bounds every tensor, so a descriptor that is not its entry is refused.
func (l *Loader) segments(shards []gguf.Shard, hashing bool) ([]*segment, error) {
	perShard := make([][]*segment, len(shards))
	for _, routed := range l.routed {
		info := routed.info
		if listed, found := l.file.Tensor(info.Name); !found || listed != info {
			return nil, fmt.Errorf("tensor %q does not match the GGUF directory", info.Name)
		}
		routed.shard = int(info.Shard)
		routed.offset = shards[routed.shard].DataOffset + info.Offset
		routed.length = info.Size
		routed.pointers = make([]driver.DevicePtr, len(routed.routes))
		perShard[routed.shard] = append(perShard[routed.shard], routed)
	}
	var result []*segment
	for index, shard := range shards {
		slices.SortFunc(perShard[index], func(left, right *segment) int { return cmp.Compare(left.offset, right.offset) })
		var cursor uint64
		for _, routed := range perShard[index] {
			if hashing && routed.offset > cursor {
				result = append(result, &segment{shard: index, offset: cursor, length: routed.offset - cursor})
			}
			result = append(result, routed)
			cursor = routed.offset + routed.length
		}
		if hashing && cursor < shard.Size {
			result = append(result, &segment{shard: index, offset: cursor, length: shard.Size - cursor})
		}
	}
	return result, nil
}

// chunk is a piece of a segment in flight between stages. Chunks of a tensor
// that converts are cut from the tensor's own storage; the rest from a pooled
// buffer the uploader returns.
type chunk struct {
	segment *segment
	offset  uint64
	data    []byte
	pooled  []byte
}

// heldTensor is an upload no store has adopted yet.
type heldTensor struct {
	store *deviceTensorStore
	name  string
	value DeviceTensor
}

type loadRun struct {
	shards   []gguf.Shard
	segments []*segment
	hashers  []*artifact.Hasher
	held     []heldTensor
}

// pipeline reads, hashes and uploads concurrently: the reader fills buffers in
// file order, the hasher sees each chunk before it moves on, and the caller
// uploads what the hasher has passed. The wall time is the slowest stage. The
// first stage to fail stops the others and names the error; a caller's
// cancellation returns the caller's own error.
func (r *loadRun) pipeline(parent context.Context, chunkBytes uint64) error {
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	failures := make(chan error, 1)
	fail := func(err error) {
		select {
		case failures <- err:
		default:
		}
		cancel()
	}
	var largest uint64
	for _, piece := range r.segments {
		if !piece.converts() {
			largest = max(largest, min(chunkBytes, piece.length))
		}
	}
	free := make(chan []byte, loadBufferCount)
	for range loadBufferCount {
		free <- make([]byte, largest)
	}
	read := make(chan chunk, loadQueueDepth)
	var stages sync.WaitGroup
	stages.Go(func() {
		defer close(read)
		if err := r.read(ctx, chunkBytes, free, read); err != nil {
			fail(err)
		}
	})
	uploads := read
	if len(r.hashers) != 0 {
		hashed := make(chan chunk, loadQueueDepth)
		stages.Go(func() {
			defer close(hashed)
			r.hash(ctx, read, hashed)
		})
		uploads = hashed
	}
	for next := range uploads {
		if err := r.upload(ctx, next); err != nil {
			fail(err)
			break
		}
		if next.pooled != nil {
			free <- next.pooled
		}
	}
	stages.Wait()
	select {
	case err := <-failures:
		return err
	default:
		return parent.Err()
	}
}

// read cuts the segments into chunks in file order. It returns nil when the
// pipeline was cancelled, which its caller reports.
func (r *loadRun) read(ctx context.Context, chunkBytes uint64, free <-chan []byte, out chan<- chunk) error {
	for _, piece := range r.segments {
		if piece.converts() {
			piece.storage = make([]byte, int(piece.length))
		}
		for offset := uint64(0); offset < piece.length; {
			length := min(chunkBytes, piece.length-offset)
			next := chunk{segment: piece, offset: offset}
			if piece.storage != nil {
				next.data = piece.storage[offset : offset+length]
			} else {
				select {
				case next.pooled = <-free:
				case <-ctx.Done():
					return nil
				}
				next.data = next.pooled[:length]
			}
			at := int64(piece.offset + offset)
			source := io.NewSectionReader(r.shards[piece.shard].Source, at, int64(len(next.data)))
			if _, err := io.ReadFull(source, next.data); err != nil {
				return fmt.Errorf("read model file %d at %d: %w", piece.shard, at, err)
			}
			select {
			case out <- next:
			case <-ctx.Done():
				return nil
			}
			offset += length
		}
	}
	return nil
}

// hash feeds each chunk to its shard's hasher, in the order the reader cut it,
// before passing it on.
func (r *loadRun) hash(ctx context.Context, in <-chan chunk, out chan<- chunk) {
	for next := range in {
		_, _ = r.hashers[next.segment.shard].Write(next.data)
		select {
		case out <- next:
		case <-ctx.Done():
			return
		}
	}
}

// upload sends a chunk of a routed tensor to its raw routes, and converts the
// tensor for its other routes once its last chunk has arrived.
func (r *loadRun) upload(ctx context.Context, next chunk) error {
	routed := next.segment
	info := routed.info
	for index, destination := range routed.routes {
		if destination.encode != nil {
			continue
		}
		if next.offset == 0 {
			pointer, err := destination.store.mem.alloc(ctx, info.Size)
			if err != nil {
				return fmt.Errorf("allocate tensor %q: %w", info.Name, err)
			}
			routed.pointers[index] = pointer
			r.held = append(r.held, heldTensor{destination.store, info.Name, DeviceTensor{Info: info, Pointer: pointer}})
		}
		if err := destination.store.mem.write(ctx, routed.pointers[index]+driver.DevicePtr(next.offset), next.data); err != nil {
			return fmt.Errorf("upload tensor %q at %d: %w", info.Name, next.offset, err)
		}
	}
	if routed.storage != nil && next.offset+uint64(len(next.data)) == info.Size {
		return r.convert(ctx, routed)
	}
	return nil
}

// convert dequantizes a tensor once, frees its file bytes, and uploads it
// through each converting route.
func (r *loadRun) convert(ctx context.Context, routed *segment) error {
	info := routed.info
	value, err := model.DecodeHostTensor(info, routed.storage)
	routed.storage = nil
	if err != nil {
		return err
	}
	for _, destination := range routed.routes {
		if destination.encode == nil {
			continue
		}
		store := destination.store
		byteCount := uint64(len(value.Data)) * destination.bytesPerValue
		encoded := destination.encode(value.Data, int(byteCount))
		if uint64(len(encoded)) != byteCount {
			return fmt.Errorf("%s tensor %q encoding size differs", store.kind, info.Name)
		}
		pointer, err := store.mem.alloc(ctx, byteCount)
		if err != nil {
			return fmt.Errorf("upload %s tensor %q: %w", store.kind, info.Name, err)
		}
		r.held = append(r.held, heldTensor{store, info.Name, DeviceTensor{Info: info, Shape: value.Shape, Pointer: pointer, Size: byteCount}})
		if err := store.mem.write(ctx, pointer, encoded); err != nil {
			return fmt.Errorf("upload %s tensor %q: %w", store.kind, info.Name, err)
		}
	}
	return nil
}

// verify compares what each shard hashed to with the identity it was claimed under.
func (r *loadRun) verify(expect []artifact.ID) error {
	for index, hasher := range r.hashers {
		id, count := hasher.ID()
		if id != expect[index] || count != r.shards[index].Size {
			return fmt.Errorf("%w: shard %d is %s over %d bytes, claimed %s", artifact.ErrContentMismatch, index, id, count, expect[index])
		}
	}
	return nil
}

// publish hands the uploads to their stores together: all of them, or none.
func (r *loadRun) publish() error {
	var stores []*deviceTensorStore
	for _, upload := range r.held {
		if !slices.Contains(stores, upload.store) {
			stores = append(stores, upload.store)
			upload.store.mu.Lock()
			defer upload.store.mu.Unlock()
		}
	}
	for _, upload := range r.held {
		if upload.store.closed {
			return fmt.Errorf("%s weights are closed", upload.store.kind)
		}
		if _, exists := upload.store.tensors[upload.name]; exists {
			return fmt.Errorf("%s tensor %q is already loaded", upload.store.kind, upload.name)
		}
	}
	for _, upload := range r.held {
		upload.store.tensors[upload.name] = upload.value
	}
	r.held = nil
	return nil
}

// release frees every upload no store adopted.
func (r *loadRun) release() error {
	var errs []error
	for _, upload := range r.held {
		errs = append(errs, upload.store.mem.free(context.Background(), []driver.DevicePtr{upload.value.Pointer}))
	}
	r.held = nil
	return errors.Join(errs...)
}
