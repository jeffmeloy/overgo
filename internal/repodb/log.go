package repodb

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"

	"llamacpp2go/internal/artifact"
)

const (
	storeFilename      = "repodb.log"
	lockFilename       = "repodb.lock"
	storeHeaderBytes   = 16
	frameHeaderBytes   = 84
	frameChecksumSize  = 4
	storeVersion       = uint16(1)
	frameVersion       = uint16(1)
	frameKindBatch     = uint16(1)
	maxFramePayload    = 64 << 20
	storeFileMode      = 0o644
	storeDirectoryMode = 0o755

	storeMagicOffset    = 0
	storeVersionOffset  = 8
	storeFlagsOffset    = 10
	storeChecksumOffset = 12

	frameMagicOffset       = 0
	frameVersionOffset     = 4
	frameKindOffset        = 6
	frameSequenceOffset    = 8
	framePreviousOffset    = 16
	frameIDOffset          = 48
	framePayloadSizeOffset = 80

	commitVersionOffset  = 0
	commitKindOffset     = 2
	commitSequenceOffset = 4
	commitPrefixBytes    = 12
)

var (
	storeMagic = [8]byte{'L', '2', 'G', 'R', 'D', 'B', '0', '1'}
	frameMagic = uint32(0x31424452)
	crcTable   = crc32.MakeTable(crc32.Castagnoli)
)

type logRecord struct {
	sequence uint64
	previous artifact.CommitID
	id       artifact.CommitID
	payload  []byte
}

type recordLog struct {
	file     *os.File
	lock     *fileLock
	readOnly bool
}

type replayResult struct {
	head      artifact.CommitID
	sequence  uint64
	validEnd  int64
	recovered bool
}

func openRecordLog(root string, readOnly bool, apply func(logRecord) error) (*recordLog, replayResult, error) {
	path := filepath.Join(root, storeFilename)
	if readOnly {
		file, err := os.Open(path)
		if err != nil {
			return nil, replayResult{}, fmt.Errorf("repodb: open read-only log: %w", err)
		}
		log := &recordLog{file: file, readOnly: true}
		result, err := log.replay(apply)
		if err != nil {
			_ = file.Close()
			return nil, replayResult{}, err
		}
		return log, result, nil
	}
	if err := os.MkdirAll(root, storeDirectoryMode); err != nil {
		return nil, replayResult{}, fmt.Errorf("repodb: create root: %w", err)
	}
	lock, err := acquireFileLock(filepath.Join(root, lockFilename))
	if err != nil {
		return nil, replayResult{}, fmt.Errorf("repodb: acquire writer lock: %w", err)
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, storeFileMode)
	if err != nil {
		_ = lock.Close()
		return nil, replayResult{}, fmt.Errorf("repodb: open log: %w", err)
	}
	log := &recordLog{file: file, lock: lock}
	if err := log.ensureHeader(); err != nil {
		_ = log.Close()
		return nil, replayResult{}, err
	}
	result, err := log.replay(apply)
	if err != nil {
		_ = log.Close()
		return nil, replayResult{}, err
	}
	if result.recovered {
		if err := file.Truncate(result.validEnd); err != nil {
			_ = log.Close()
			return nil, replayResult{}, fmt.Errorf("repodb: truncate torn tail: %w", err)
		}
		if err := file.Sync(); err != nil {
			_ = log.Close()
			return nil, replayResult{}, fmt.Errorf("repodb: sync recovered log: %w", err)
		}
	}
	if _, err := file.Seek(0, io.SeekEnd); err != nil {
		_ = log.Close()
		return nil, replayResult{}, fmt.Errorf("repodb: seek log end: %w", err)
	}
	return log, result, nil
}

func (l *recordLog) ensureHeader() error {
	info, err := l.file.Stat()
	if err != nil {
		return fmt.Errorf("repodb: stat log: %w", err)
	}
	if info.Size() != 0 {
		return nil
	}
	header := encodeStoreHeader()
	written, err := l.file.Write(header)
	if err != nil || written != len(header) {
		if err == nil {
			err = io.ErrShortWrite
		}
		return fmt.Errorf("repodb: write header: %w", err)
	}
	if err := l.file.Sync(); err != nil {
		return fmt.Errorf("repodb: sync header: %w", err)
	}
	return nil
}

func encodeStoreHeader() []byte {
	header := make([]byte, storeHeaderBytes)
	copy(header[storeMagicOffset:storeVersionOffset], storeMagic[:])
	binary.LittleEndian.PutUint16(header[storeVersionOffset:storeFlagsOffset], storeVersion)
	binary.LittleEndian.PutUint16(header[storeFlagsOffset:storeChecksumOffset], 0)
	binary.LittleEndian.PutUint32(header[storeChecksumOffset:storeHeaderBytes], crc32.Checksum(header[:storeChecksumOffset], crcTable))
	return header
}

func validateStoreHeader(header []byte) error {
	if len(header) != storeHeaderBytes || string(header[storeMagicOffset:storeVersionOffset]) != string(storeMagic[:]) {
		return errors.New("repodb: invalid log magic")
	}
	if binary.LittleEndian.Uint16(header[storeVersionOffset:storeFlagsOffset]) != storeVersion {
		return errors.New("repodb: unsupported log version")
	}
	if binary.LittleEndian.Uint16(header[storeFlagsOffset:storeChecksumOffset]) != 0 {
		return errors.New("repodb: unsupported log flags")
	}
	if binary.LittleEndian.Uint32(header[storeChecksumOffset:storeHeaderBytes]) != crc32.Checksum(header[:storeChecksumOffset], crcTable) {
		return errors.New("repodb: invalid log header checksum")
	}
	return nil
}

func (l *recordLog) replay(apply func(logRecord) error) (replayResult, error) {
	if _, err := l.file.Seek(0, io.SeekStart); err != nil {
		return replayResult{}, fmt.Errorf("repodb: seek log start: %w", err)
	}
	header := make([]byte, storeHeaderBytes)
	if _, err := io.ReadFull(l.file, header); err != nil {
		return replayResult{}, fmt.Errorf("repodb: read log header: %w", err)
	}
	if err := validateStoreHeader(header); err != nil {
		return replayResult{}, err
	}
	result := replayResult{validEnd: storeHeaderBytes}
	for {
		frameStart := result.validEnd
		frameHeader := make([]byte, frameHeaderBytes)
		count, err := io.ReadFull(l.file, frameHeader)
		if errors.Is(err, io.EOF) && count == 0 {
			break
		}
		if err != nil {
			result.recovered = true
			break
		}
		payloadSize, record, err := decodeFrameHeader(frameHeader)
		if err != nil {
			return replayResult{}, fmt.Errorf("repodb: frame at %d: %w", frameStart, err)
		}
		if payloadSize > maxFramePayload {
			return replayResult{}, fmt.Errorf("repodb: frame at %d exceeds payload limit", frameStart)
		}
		body := make([]byte, int(payloadSize)+frameChecksumSize)
		if _, err := io.ReadFull(l.file, body); err != nil {
			result.recovered = true
			break
		}
		record.payload = body[:payloadSize]
		checksum := binary.LittleEndian.Uint32(body[payloadSize:])
		actual := crc32.New(crcTable)
		_, _ = actual.Write(frameHeader)
		_, _ = actual.Write(record.payload)
		if checksum != actual.Sum32() {
			return replayResult{}, fmt.Errorf("repodb: frame at %d has invalid checksum", frameStart)
		}
		if record.sequence != result.sequence+1 || record.previous != result.head {
			return replayResult{}, fmt.Errorf("repodb: frame at %d breaks commit chain", frameStart)
		}
		if record.id != commitIdentity(record.sequence, record.previous, record.payload) {
			return replayResult{}, fmt.Errorf("repodb: frame at %d has invalid commit identity", frameStart)
		}
		if err := apply(record); err != nil {
			return replayResult{}, fmt.Errorf("repodb: apply frame at %d: %w", frameStart, err)
		}
		result.sequence = record.sequence
		result.head = record.id
		result.validEnd += int64(frameHeaderBytes) + int64(payloadSize) + frameChecksumSize
	}
	return result, nil
}

func decodeFrameHeader(header []byte) (uint32, logRecord, error) {
	if binary.LittleEndian.Uint32(header[frameMagicOffset:frameVersionOffset]) != frameMagic {
		return 0, logRecord{}, errors.New("invalid frame magic")
	}
	if binary.LittleEndian.Uint16(header[frameVersionOffset:frameKindOffset]) != frameVersion {
		return 0, logRecord{}, errors.New("unsupported frame version")
	}
	if binary.LittleEndian.Uint16(header[frameKindOffset:frameSequenceOffset]) != frameKindBatch {
		return 0, logRecord{}, errors.New("unsupported frame kind")
	}
	record := logRecord{sequence: binary.LittleEndian.Uint64(header[frameSequenceOffset:framePreviousOffset])}
	copy(record.previous[:], header[framePreviousOffset:frameIDOffset])
	copy(record.id[:], header[frameIDOffset:framePayloadSizeOffset])
	return binary.LittleEndian.Uint32(header[framePayloadSizeOffset:frameHeaderBytes]), record, nil
}

func commitIdentity(sequence uint64, previous artifact.CommitID, payload []byte) artifact.CommitID {
	hasher := sha256.New()
	var prefix [commitPrefixBytes]byte
	binary.LittleEndian.PutUint16(prefix[commitVersionOffset:commitKindOffset], frameVersion)
	binary.LittleEndian.PutUint16(prefix[commitKindOffset:commitSequenceOffset], frameKindBatch)
	binary.LittleEndian.PutUint64(prefix[commitSequenceOffset:commitPrefixBytes], sequence)
	_, _ = hasher.Write(prefix[:])
	_, _ = hasher.Write(previous[:])
	_, _ = hasher.Write(payload)
	var id artifact.CommitID
	copy(id[:], hasher.Sum(nil))
	return id
}

func encodeRecord(sequence uint64, previous artifact.CommitID, payload []byte) (artifact.CommitID, []byte) {
	id := commitIdentity(sequence, previous, payload)
	header := make([]byte, frameHeaderBytes)
	binary.LittleEndian.PutUint32(header[frameMagicOffset:frameVersionOffset], frameMagic)
	binary.LittleEndian.PutUint16(header[frameVersionOffset:frameKindOffset], frameVersion)
	binary.LittleEndian.PutUint16(header[frameKindOffset:frameSequenceOffset], frameKindBatch)
	binary.LittleEndian.PutUint64(header[frameSequenceOffset:framePreviousOffset], sequence)
	copy(header[framePreviousOffset:frameIDOffset], previous[:])
	copy(header[frameIDOffset:framePayloadSizeOffset], id[:])
	binary.LittleEndian.PutUint32(header[framePayloadSizeOffset:frameHeaderBytes], uint32(len(payload)))
	frame := make([]byte, 0, len(header)+len(payload)+frameChecksumSize)
	frame = append(frame, header...)
	frame = append(frame, payload...)
	checksum := crc32.Checksum(frame, crcTable)
	frame = binary.LittleEndian.AppendUint32(frame, checksum)
	return id, frame
}

func (l *recordLog) append(sequence uint64, previous artifact.CommitID, payload []byte) (artifact.CommitID, error) {
	if l == nil || l.file == nil || l.readOnly {
		return artifact.CommitID{}, errors.New("repodb: log is not writable")
	}
	if len(payload) > maxFramePayload {
		return artifact.CommitID{}, errors.New("repodb: batch exceeds payload limit")
	}
	id, frame := encodeRecord(sequence, previous, payload)
	written, err := l.file.Write(frame)
	if err != nil || written != len(frame) {
		if err == nil {
			err = io.ErrShortWrite
		}
		return id, fmt.Errorf("repodb: append commit: %w", err)
	}
	if err := l.file.Sync(); err != nil {
		return id, fmt.Errorf("repodb: sync commit: %w", err)
	}
	return id, nil
}

func (l *recordLog) Close() error {
	if l == nil {
		return nil
	}
	var result error
	if l.file != nil {
		result = l.file.Close()
		l.file = nil
	}
	if l.lock != nil {
		if err := l.lock.Close(); result == nil {
			result = err
		}
		l.lock = nil
	}
	return result
}
