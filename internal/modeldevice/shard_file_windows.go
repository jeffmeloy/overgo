//go:build windows

package modeldevice

import (
	"errors"
	"os"
	"syscall"
)

const (
	// fileFlagNoBuffering is FILE_FLAG_NO_BUFFERING: a read moves bytes from
	// the disk into the caller's buffer without the file cache's copy, at the
	// rate the drive gives several reads in flight.
	fileFlagNoBuffering = 0x20000000
	// unbufferedAlignment bounds an unbuffered read's offset, length and buffer
	// address: a multiple of the sector size of the volume, which is 512 or
	// 4096 bytes on the drives Windows reports, and of the page size.
	unbufferedAlignment = 4096
	// overlappedHighShift splits a file offset into OVERLAPPED's two 32-bit halves.
	overlappedHighShift = 32
)

// unbufferedFile reads a shard past the file cache.
type unbufferedFile struct{ handle syscall.Handle }

// openShardFile opens a shard for unbuffered reads, or through the file cache
// where the file system refuses them.
func openShardFile(path string) (shardFile, uint64, error) {
	name, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return nil, 0, err
	}
	handle, err := syscall.CreateFile(name, syscall.GENERIC_READ, syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE, nil,
		syscall.OPEN_EXISTING, syscall.FILE_ATTRIBUTE_NORMAL|fileFlagNoBuffering, 0)
	if err == nil {
		return unbufferedFile{handle}, unbufferedAlignment, nil
	}
	file, openErr := os.Open(path)
	if openErr != nil {
		return nil, 0, errors.Join(err, openErr)
	}
	return readerAtFile{source: file, owned: file}, 1, nil
}

func (f unbufferedFile) readAt(buffer []byte, offset uint64) (int, error) {
	read := 0
	for read < len(buffer) {
		at := offset + uint64(read)
		overlapped := syscall.Overlapped{Offset: uint32(at), OffsetHigh: uint32(at >> overlappedHighShift)}
		var done uint32
		err := syscall.ReadFile(f.handle, buffer[read:], &done, &overlapped)
		if errors.Is(err, syscall.ERROR_HANDLE_EOF) {
			break
		}
		if err != nil {
			return read, err
		}
		read += int(done)
		// Only the shard's end returns part of a sector.
		if done == 0 || done%unbufferedAlignment != 0 {
			break
		}
	}
	return read, nil
}

func (f unbufferedFile) close() error { return syscall.CloseHandle(f.handle) }
