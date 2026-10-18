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
	"overgo/internal/checked"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor"
)

const (
	// loadReaders is the reads a shard's pass keeps in flight, each through a
	// handle of its own: a drive reaches its rate only with several reads
	// queued, and one handle serializes them.
	loadReaders = 4
	// loadReaderBuffers is the blocks one reader holds: the one it fills while
	// the hash or the upload holds the one it filled before.
	loadReaderBuffers = 2
)

// Loader reads a model file once and uploads each routed tensor to the stores
// it is routed to. Shards are read concurrently, each in file order through
// its own readers, hasher and uploader. Given the identity every shard must
// carry, the same pass feeds every byte of every shard, header, padding and
// unrouted tensors included, to that shard's hasher before the byte is
// uploaded, converted or placed in the expert residency; it publishes nothing,
// and frees every allocation, unless every digest matches. A Loader runs once.
type Loader struct {
	file       *gguf.File
	chunkBytes uint64
	routed     []*segment
	byName     map[string]*segment
	experts    *expertRoute
	open       func(path string) (shardFile, uint64, error)
	err        error
}

// expertRoute is the banks a pass places in a store's expert residency, and
// the card bytes the residency may hold; zero takes the free memory, read
// once every other route holds its allocation, less expertFillHeadroomBytes.
type expertRoute struct {
	store *DeviceWeights
	infos []gguf.TensorInfo
	limit uint64
}

// segment is a routed tensor: where its bytes lie in its shard and where
// they go.
type segment struct {
	shard  int
	offset uint64
	length uint64

	info     gguf.TensorInfo
	routes   []route
	pointers []driver.DevicePtr // each route's allocation, made before the pass reads
	storage  []byte             // the whole tensor, gathered for routes that convert
	bank     *expertBank        // the residency's bank, for a tensor routed to experts
}

// route is one destination of a tensor's bytes: a raw device copy, a
// converted one, or the slots of an expert bank.
type route struct {
	store         *deviceTensorStore
	encode        convertedTensorEncoder // nil: the bytes upload as stored
	bytesPerValue uint64
	expert        bool
}

// NewLoader starts a pass over the file with no tensors routed.
func NewLoader(file *gguf.File) *Loader {
	return &Loader{file: file, chunkBytes: model.WeightChunkSize, byName: make(map[string]*segment), open: openShardFile}
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

// Experts routes routed-expert banks into the store's expert residency: the
// pass copies each expert, from the bytes it reads and hashes, to a card slot
// while limit allows and to pinned host memory the card maps otherwise, and
// the store serves the banks from the tables the pass publishes.
func (l *Loader) Experts(store *DeviceWeights, infos []gguf.TensorInfo, limit uint64) {
	if l.experts != nil {
		l.err = errors.New("expert banks are routed twice")
		return
	}
	l.experts = &expertRoute{store: store, infos: infos, limit: limit}
	for _, info := range infos {
		l.route(info, route{store: store.deviceTensorStore, expert: true})
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
	run := &loadRun{shards: shards, paths: l.file.SourcePaths(), open: l.open}
	var err error
	if run.segments, err = l.segments(shards); err != nil {
		return err
	}
	for _, id := range expect {
		hasher, err := artifact.NewHasher(id.Kind())
		if err != nil {
			return err
		}
		run.hashers = append(run.hashers, hasher)
	}
	err = run.allocate(ctx)
	if err == nil && l.experts != nil {
		err = run.prepareExperts(ctx, l.file, l.experts)
	}
	if err == nil {
		err = run.pipeline(ctx, l.chunkBytes)
	}
	if err == nil {
		err = run.verify(expect)
	}
	if err == nil {
		err = run.publish(ctx)
	}
	if err != nil {
		return errors.Join(err, run.release())
	}
	return nil
}

// segments lists each shard's routed tensors in file order. The directory
// the file was parsed from bounds every tensor, so a descriptor that is not
// its entry is refused.
func (l *Loader) segments(shards []gguf.Shard) ([][]*segment, error) {
	perShard := make([][]*segment, len(shards))
	for _, routed := range l.routed {
		info := routed.info
		if listed, found := l.file.Tensor(info.Name); !found || listed != info {
			return nil, fmt.Errorf("tensor %q does not match the GGUF directory", info.Name)
		}
		if len(routed.routes) > 1 && slices.ContainsFunc(routed.routes, func(r route) bool { return r.expert }) {
			return nil, fmt.Errorf("expert bank %q takes another route", info.Name)
		}
		routed.shard = int(info.Shard)
		routed.offset = shards[routed.shard].DataOffset + info.Offset
		routed.length = info.Size
		routed.pointers = make([]driver.DevicePtr, len(routed.routes))
		perShard[routed.shard] = append(perShard[routed.shard], routed)
	}
	for _, routed := range perShard {
		slices.SortFunc(routed, func(left, right *segment) int { return cmp.Compare(left.offset, right.offset) })
	}
	return perShard, nil
}

// heldTensor is an upload no store has adopted yet.
type heldTensor struct {
	store *deviceTensorStore
	name  string
	value DeviceTensor
}

type loadRun struct {
	shards   []gguf.Shard
	paths    []string
	open     func(path string) (shardFile, uint64, error)
	segments [][]*segment
	hashers  []*artifact.Hasher
	held     []heldTensor
	// experts is the residency the pass fills, owned by expertStore until it publishes.
	experts     *expertResidency
	expertStore *DeviceWeights
}

// allocate makes every upload's device memory before the pass reads a byte,
// so an expert residency measures the card against what the rest will hold.
func (r *loadRun) allocate(ctx context.Context) error {
	for _, routed := range r.segments {
		for _, piece := range routed {
			info := piece.info
			for index, destination := range piece.routes {
				if destination.expert {
					continue
				}
				value := DeviceTensor{Info: info}
				size := info.Size
				if destination.encode != nil {
					shape, err := tensor.NewShape(info.Extents()...)
					if err != nil {
						return fmt.Errorf("%s tensor %q shape: %w", destination.store.kind, info.Name, err)
					}
					elements, err := shape.Elements()
					bytes, fits := checked.Bytes(elements, destination.bytesPerValue)
					if err != nil || !fits {
						return fmt.Errorf("%s tensor %q size overflows", destination.store.kind, info.Name)
					}
					value.Shape, value.Size, size = shape, bytes, bytes
				}
				pointer, err := destination.store.mem.alloc(ctx, size)
				if err != nil {
					return fmt.Errorf("allocate %s tensor %q: %w", destination.store.kind, info.Name, err)
				}
				value.Pointer, piece.pointers[index] = pointer, pointer
				r.held = append(r.held, heldTensor{destination.store, info.Name, value})
			}
		}
	}
	return nil
}

// prepareExperts allocates the residency the pass fills, on the store's
// worker, and names each routed bank's place in it.
func (r *loadRun) prepareExperts(ctx context.Context, file *gguf.File, experts *expertRoute) error {
	r.expertStore = experts.store
	return experts.store.onDevice(ctx, func(state *device.State, mem expertMemory) error {
		if experts.store.experts != nil {
			return errors.New("expert residency is already configured")
		}
		limit := experts.limit
		if limit == 0 {
			free, _, err := state.Driver.MemInfo()
			if err != nil {
				return err
			}
			limit = free - min(free, expertFillHeadroomBytes)
		}
		residency, err := newExpertResidency(ctx, file, experts.infos, limit, mem)
		if err != nil {
			return err
		}
		residency.state, r.experts = state, residency
		for _, routed := range r.segments {
			for _, piece := range routed {
				piece.bank = residency.banks[piece.info.Name]
			}
		}
		return nil
	})
}

// block is one read of a shard: length bytes from start into buffer, of
// which [from, to) belong to the pass.
type block struct {
	start, length, from, to uint64
	buffer                  []byte
	reader                  int
}

// bytes are the block's bytes that belong to the pass.
func (b block) bytes() []byte { return b.buffer[b.from-b.start : b.to-b.start] }

// pipeline runs every shard's pass concurrently. The first stage to fail
// stops the others and names the error; a caller's cancellation returns the
// caller's own error.
func (r *loadRun) pipeline(parent context.Context, blockBytes uint64) error {
	ctx, cancel := context.WithCancelCause(parent)
	defer cancel(nil)
	var passes sync.WaitGroup
	for shard := range r.shards {
		passes.Go(func() {
			if err := r.pass(ctx, shard, blockBytes); err != nil {
				cancel(err)
			}
		})
	}
	passes.Wait()
	return context.Cause(ctx)
}

// pass reads one shard in blocks, loadReaders at a time, hashes the blocks
// in file order and hands each routed tensor's bytes in them to its routes.
// It reads the whole shard when it hashes, and only the routed tensors
// otherwise.
func (r *loadRun) pass(parent context.Context, shard int, blockBytes uint64) error {
	routed := r.segments[shard]
	hashing := len(r.hashers) != 0
	if !hashing && len(routed) == 0 {
		return nil
	}
	files, align, err := r.openShard(shard)
	if err != nil {
		return err
	}
	blocks, largest, err := r.blocks(shard, blockBytes, align)
	if err != nil {
		return errors.Join(err, closeShard(files))
	}
	ctx, cancel := context.WithCancelCause(parent)
	defer cancel(nil)
	var stages sync.WaitGroup
	filled := make([]chan block, min(len(files), len(blocks)))
	free := make([]chan []byte, len(filled))
	for reader := range filled {
		filled[reader], free[reader] = make(chan block, loadReaderBuffers), make(chan []byte, loadReaderBuffers)
		for range loadReaderBuffers {
			free[reader] <- alignedBuffer(largest, align)
		}
		stages.Go(func() {
			if err := r.read(ctx, shard, files[reader], blocks, reader, len(filled), free[reader], filled[reader]); err != nil {
				cancel(err)
			}
		})
	}
	ordered := make(chan block, loadReaderBuffers)
	stages.Go(func() {
		defer close(ordered)
		r.order(ctx, shard, len(blocks), filled, ordered)
	})
	cursor := 0
	for next := range ordered {
		if err := r.dispatch(ctx, routed, &cursor, next); err != nil {
			cancel(err)
			break
		}
		free[next.reader] <- next.buffer
	}
	stages.Wait()
	return errors.Join(context.Cause(ctx), closeShard(files))
}

// openShard opens a shard once for each reader: by its path, so that each
// read has its own handle, or through the source a file parsed from a reader
// was given. align is the boundary every read keeps.
func (r *loadRun) openShard(shard int) ([]shardFile, uint64, error) {
	files := make([]shardFile, 0, loadReaders)
	align := uint64(1)
	for range loadReaders {
		if len(r.paths) == 0 {
			files = append(files, readerAtFile{source: r.shards[shard].Source})
			continue
		}
		file, fileAlign, err := r.open(r.paths[shard])
		if err != nil {
			return nil, 0, errors.Join(err, closeShard(files))
		}
		files, align = append(files, file), max(align, fileAlign)
	}
	return files, align, nil
}

func closeShard(files []shardFile) error {
	var errs []error
	for _, file := range files {
		errs = append(errs, file.close())
	}
	return errors.Join(errs...)
}

// byteRange is the bytes from start up to end of a shard.
type byteRange struct{ start, end uint64 }

// blocks cuts the ranges of the shard the pass reads into aligned reads of
// at most blockBytes: the whole shard when it hashes, else each routed
// tensor. It returns the longest read.
func (r *loadRun) blocks(shard int, blockBytes, align uint64) ([]block, uint64, error) {
	extents := []byteRange{{0, r.shards[shard].Size}}
	if len(r.hashers) == 0 {
		extents = nil
		for _, piece := range r.segments[shard] {
			extents = append(extents, byteRange{piece.offset, piece.offset + piece.length})
		}
	}
	step, ok := checked.RoundUpMultiple(blockBytes, align)
	var blocks []block
	var largest uint64
	for _, extent := range extents {
		end, fits := checked.RoundUpMultiple(extent.end, align)
		if !ok || !fits {
			return nil, 0, errors.New("model load: shard read range overflows")
		}
		for at := extent.start - extent.start%align; at < end; at += step {
			length := min(step, end-at)
			blocks = append(blocks, block{start: at, length: length, from: max(at, extent.start), to: min(at+length, extent.end)})
			largest = max(largest, length)
		}
	}
	return blocks, largest, nil
}

// read fills every count-th block from first, in order, and passes each on.
// It returns nil when the pass was cancelled, which its caller reports.
func (r *loadRun) read(ctx context.Context, shard int, file shardFile, blocks []block, first, count int, free <-chan []byte, filled chan<- block) error {
	for index := first; index < len(blocks); index += count {
		next := blocks[index]
		select {
		case next.buffer = <-free:
		case <-ctx.Done():
			return nil
		}
		n, err := file.readAt(next.buffer[:next.length], next.start)
		if err == nil && uint64(n) < next.to-next.start {
			err = io.ErrUnexpectedEOF
		}
		if err != nil {
			return fmt.Errorf("read model file %d at %d: %w", shard, next.start, err)
		}
		next.reader = first
		select {
		case filled <- next:
		case <-ctx.Done():
			return nil
		}
	}
	return nil
}

// order passes the blocks on in file order, each through its shard's hasher
// first.
func (r *loadRun) order(ctx context.Context, shard, count int, filled []chan block, out chan<- block) {
	for index := range count {
		var next block
		select {
		case next = <-filled[index%len(filled)]:
		case <-ctx.Done():
			return
		}
		if len(r.hashers) != 0 {
			_, _ = r.hashers[shard].Write(next.bytes())
		}
		select {
		case out <- next:
		case <-ctx.Done():
			return
		}
	}
}

// dispatch hands the bytes of the routed tensors a block holds to their
// routes; cursor is the first tensor of the shard not yet complete.
func (r *loadRun) dispatch(ctx context.Context, routed []*segment, cursor *int, next block) error {
	for ; *cursor < len(routed); *cursor++ {
		piece := routed[*cursor]
		end := piece.offset + piece.length
		if piece.offset >= next.to {
			return nil
		}
		if from, to := max(piece.offset, next.from), min(end, next.to); from < to {
			if err := r.upload(ctx, piece, from-piece.offset, next.buffer[from-next.start:to-next.start]); err != nil {
				return err
			}
		}
		if end > next.to {
			return nil
		}
	}
	return nil
}

// upload sends bytes of a routed tensor, offset bytes into it, to its raw
// routes or its expert slots, and converts the tensor for its other routes
// once its last bytes have arrived.
func (r *loadRun) upload(ctx context.Context, piece *segment, offset uint64, data []byte) error {
	info := piece.info
	if piece.bank != nil {
		return r.experts.place(ctx, r.expertStore.onDevice, piece.bank, offset, data)
	}
	for index, destination := range piece.routes {
		if destination.encode != nil {
			continue
		}
		if err := destination.store.mem.write(ctx, piece.pointers[index]+driver.DevicePtr(offset), data); err != nil {
			return fmt.Errorf("upload tensor %q at %d: %w", info.Name, offset, err)
		}
	}
	if !piece.converts() {
		return nil
	}
	if piece.storage == nil {
		piece.storage = make([]byte, piece.length)
	}
	copy(piece.storage[offset:], data)
	if offset+uint64(len(data)) == piece.length {
		return r.convert(ctx, piece)
	}
	return nil
}

// convert dequantizes a tensor once, frees its file bytes, and uploads it
// through each converting route.
func (r *loadRun) convert(ctx context.Context, piece *segment) error {
	info := piece.info
	value, err := model.DecodeHostTensor(info, piece.storage)
	piece.storage = nil
	if err != nil {
		return err
	}
	for index, destination := range piece.routes {
		if destination.encode == nil {
			continue
		}
		store := destination.store
		byteCount := uint64(len(value.Data)) * destination.bytesPerValue
		encoded := destination.encode(value.Data, int(byteCount))
		if uint64(len(encoded)) != byteCount {
			return fmt.Errorf("%s tensor %q encoding size differs", store.kind, info.Name)
		}
		if err := store.mem.write(ctx, piece.pointers[index], encoded); err != nil {
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

// publish hands the uploads to their stores and the expert residency to its
// store together, its tables written first: all of them, or none.
func (r *loadRun) publish(ctx context.Context) error {
	if r.experts != nil {
		if err := r.expertStore.onDevice(ctx, func(*device.State, expertMemory) error { return r.experts.publish(ctx) }); err != nil {
			return err
		}
	}
	var stores []*deviceTensorStore
	lock := func(store *deviceTensorStore) {
		if !slices.Contains(stores, store) {
			stores = append(stores, store)
			store.mu.Lock()
		}
	}
	defer func() {
		for _, store := range stores {
			store.mu.Unlock()
		}
	}()
	for _, upload := range r.held {
		lock(upload.store)
	}
	if r.experts != nil {
		lock(r.expertStore.deviceTensorStore)
	}
	for _, store := range stores {
		if store.closed {
			return fmt.Errorf("%s weights are closed", store.kind)
		}
	}
	for _, upload := range r.held {
		if _, exists := upload.store.tensors[upload.name]; exists {
			return fmt.Errorf("%s tensor %q is already loaded", upload.store.kind, upload.name)
		}
	}
	for _, upload := range r.held {
		upload.store.tensors[upload.name] = upload.value
	}
	r.held = nil
	if r.experts != nil {
		r.expertStore.experts, r.experts = r.experts, nil
	}
	return nil
}

// release frees every upload no store adopted, and the residency its store
// did not adopt.
func (r *loadRun) release() error {
	var errs []error
	for _, upload := range r.held {
		errs = append(errs, upload.store.mem.free(context.Background(), []driver.DevicePtr{upload.value.Pointer}))
	}
	r.held = nil
	if residency := r.experts; residency != nil {
		r.experts = nil
		errs = append(errs, r.expertStore.onDevice(context.Background(), func(*device.State, expertMemory) error { return residency.close() }))
	}
	return errors.Join(errs...)
}
