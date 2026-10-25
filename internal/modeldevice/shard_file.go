package modeldevice

import (
	"errors"
	"io"
	"unsafe"
)

// shardFile reads one shard of a model file for a load pass. A read starts
// at an offset and fills a buffer whose address, offset and length are
// multiples of the alignment the file was opened with; it returns fewer bytes
// only where the shard ends.
type shardFile interface {
	readAt(buffer []byte, offset uint64) (int, error)
	close() error
}

// readerAtFile reads through an io.ReaderAt, which needs no alignment; it
// closes the source only when the pass opened it.
type readerAtFile struct {
	source io.ReaderAt
	owned  io.Closer
}

func (f readerAtFile) readAt(buffer []byte, offset uint64) (int, error) {
	n, err := f.source.ReadAt(buffer, int64(offset))
	if errors.Is(err, io.EOF) {
		err = nil
	}
	return n, err
}

func (f readerAtFile) close() error {
	if f.owned == nil {
		return nil
	}
	return f.owned.Close()
}

// alignedBuffer returns length bytes whose address is a multiple of align.
func alignedBuffer(length, align uint64) []byte {
	raw := make([]byte, length+align-1)
	shift := (align - uint64(uintptr(unsafe.Pointer(unsafe.SliceData(raw))))%align) % align
	return raw[shift : shift+length : shift+length]
}
