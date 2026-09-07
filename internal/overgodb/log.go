package overgodb

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/artifact"
)

const (
	segmentDirectory   = "segments"
	segmentExtension   = ".segment"
	storeFilename      = "overgodb.log"
	lockFilename       = "overgodb.lock"
	storeHeaderBytes   = 16
	frameHeaderBytes   = 84
	storeVersion       = uint16(1)
	frameVersion       = uint16(4)
	frameKindBatch     = uint16(1)
	maxFramePayload    = artifact.MaxContentBytes
	storeFileMode      = 0o644
	storeDirectoryMode = 0o755
	activeSegment      = uint64(0)
	segmentNameRadix   = 10
	segmentNameBitSize = 64

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
	version  uint16
	sequence uint64
	previous artifact.CommitID
	id       artifact.CommitID
	segment  uint64
	offset   int64
	payload  []byte
}

type recordLog struct {
	file         *os.File
	writer       durableWriter
	segment      uint64
	readOnly     bool
	sealed       []string
	pendingReset bool
}

type durableWriter interface {
	io.Writer
	Sync() error
}

type replayResult struct {
	head      artifact.CommitID
	sequence  uint64
	validEnd  int64
	recovered bool
}

type replayAnchor struct {
	sequence uint64
	head     artifact.CommitID
	offset   int64
	digest   [sha256.Size]byte
}

func sealedSegmentPaths(root string) ([]string, error) {
	paths, err := filepath.Glob(filepath.Join(root, segmentDirectory, "*"+segmentExtension))
	if err != nil {
		return nil, fmt.Errorf("overgodb: list segments: %w", err)
	}
	slices.Sort(paths)
	return paths, nil
}

// replaySealedSegments replays every immutable segment in order,
// threading the commit chain across files. A torn or corrupt sealed
// segment refuses: only the active segment may recover a tail.
func replaySealedSegments(paths []string, apply func(logRecord) error) (replayResult, error) {
	chain := replayResult{}
	for _, path := range paths {
		base := strings.TrimSuffix(filepath.Base(path), segmentExtension)
		segmentEnd, parseErr := strconv.ParseUint(base, segmentNameRadix, segmentNameBitSize)
		if parseErr != nil || segmentEnd == 0 {
			return replayResult{}, fmt.Errorf("overgodb: invalid sealed segment name %s", filepath.Base(path))
		}
		file, err := os.Open(path)
		if err != nil {
			return replayResult{}, fmt.Errorf("overgodb: open sealed segment: %w", err)
		}
		segment := &recordLog{file: file, segment: segmentEnd, readOnly: true}
		result, err := segment.replaySealed(chain, apply)
		_ = file.Close()
		if err != nil {
			return replayResult{}, fmt.Errorf("overgodb: sealed segment %s: %w", filepath.Base(path), err)
		}
		if result.recovered {
			return replayResult{}, fmt.Errorf("overgodb: sealed segment %s is torn", filepath.Base(path))
		}
		chain = result
	}
	return chain, nil
}

// replaySealed replays one immutable segment file, continuing the
// cross-segment commit chain the caller threads.
func (l *recordLog) replaySealed(chain replayResult, apply func(logRecord) error) (replayResult, error) {
	if _, err := l.file.Seek(0, io.SeekStart); err != nil {
		return replayResult{}, fmt.Errorf("overgodb: seek sealed segment: %w", err)
	}
	header := make([]byte, storeHeaderBytes)
	if _, err := io.ReadFull(l.file, header); err != nil {
		return replayResult{}, fmt.Errorf("overgodb: read sealed header: %w", err)
	}
	if err := validateStoreHeader(header); err != nil {
		return replayResult{}, err
	}
	result := replayResult{head: chain.head, sequence: chain.sequence, validEnd: storeHeaderBytes}
	return l.replayFrames(result, replayAnchor{}, apply)
}

func openRecordLog(
	root string,
	readOnly bool,
	anchor replayAnchor,
	apply func(logRecord) error,
) (*recordLog, replayResult, error) {
	path := filepath.Join(root, storeFilename)
	paths, err := sealedSegmentPaths(root)
	if err != nil {
		return nil, replayResult{}, err
	}
	chain := replayResult{}
	if anchor.sequence == 0 {
		sealed, err := replaySealedSegments(paths, apply)
		if err != nil {
			return nil, replayResult{}, err
		}
		chain = sealed
	}
	if readOnly {
		file, err := os.Open(path)
		if err != nil {
			return nil, replayResult{}, fmt.Errorf("overgodb: open read-only log: %w", err)
		}
		log := &recordLog{file: file, readOnly: true, sealed: paths}
		result, err := log.replayActive(chain, anchor, apply)
		if err != nil {
			_ = file.Close()
			return nil, replayResult{}, err
		}
		return log, result, nil
	}
	if err := os.MkdirAll(root, storeDirectoryMode); err != nil {
		return nil, replayResult{}, fmt.Errorf("overgodb: create root: %w", err)
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, storeFileMode)
	if err != nil {
		return nil, replayResult{}, fmt.Errorf("overgodb: open log: %w", err)
	}
	log := &recordLog{file: file, sealed: paths}
	if err := log.ensureHeader(); err != nil {
		_ = log.Close()
		return nil, replayResult{}, err
	}
	result, err := log.replayActive(chain, anchor, apply)
	if err != nil {
		_ = log.Close()
		return nil, replayResult{}, err
	}
	if result.recovered {
		if err := file.Truncate(result.validEnd); err != nil {
			_ = log.Close()
			return nil, replayResult{}, fmt.Errorf("overgodb: truncate torn tail: %w", err)
		}
		if err := file.Sync(); err != nil {
			_ = log.Close()
			return nil, replayResult{}, fmt.Errorf("overgodb: sync recovered log: %w", err)
		}
	}
	if _, err := file.Seek(0, io.SeekEnd); err != nil {
		_ = log.Close()
		return nil, replayResult{}, fmt.Errorf("overgodb: seek log end: %w", err)
	}
	return log, result, nil
}

func (l *recordLog) ensureHeader() error {
	info, err := l.file.Stat()
	if err != nil {
		return fmt.Errorf("overgodb: stat log: %w", err)
	}
	if info.Size() != 0 {
		return nil
	}
	header := encodeStoreHeader()
	if err := writeAll(l.file, header); err != nil {
		return fmt.Errorf("overgodb: write header: %w", err)
	}
	if err := l.file.Sync(); err != nil {
		return fmt.Errorf("overgodb: sync header: %w", err)
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
		return errors.New("overgodb: invalid log magic")
	}
	if binary.LittleEndian.Uint16(header[storeVersionOffset:storeFlagsOffset]) != storeVersion {
		return errors.New("overgodb: unsupported log version")
	}
	if binary.LittleEndian.Uint16(header[storeFlagsOffset:storeChecksumOffset]) != 0 {
		return errors.New("overgodb: unsupported log flags")
	}
	if binary.LittleEndian.Uint32(header[storeChecksumOffset:storeHeaderBytes]) != crc32.Checksum(header[:storeChecksumOffset], crcTable) {
		return errors.New("overgodb: invalid log header checksum")
	}
	return nil
}

// replayActive replays the active segment, continuing the sealed
// chain when the whole journal is replayed and honoring a snapshot or
// checkpoint anchor -- which always addresses the active segment --
// otherwise.
func (l *recordLog) replayActive(chain replayResult, anchor replayAnchor, apply func(logRecord) error) (replayResult, error) {
	pending, err := l.pendingRotation()
	if err != nil {
		return replayResult{}, err
	}
	if pending {
		if anchor.sequence != 0 {
			return replayResult{}, ErrSnapshotAnchor
		}
		return l.finishRotation(chain)
	}
	if anchor.sequence != 0 {
		return l.replay(anchor, apply)
	}
	if _, err := l.file.Seek(0, io.SeekStart); err != nil {
		return replayResult{}, fmt.Errorf("overgodb: seek log start: %w", err)
	}
	header := make([]byte, storeHeaderBytes)
	if _, err := io.ReadFull(l.file, header); err != nil {
		return replayResult{}, fmt.Errorf("overgodb: read log header: %w", err)
	}
	if err := validateStoreHeader(header); err != nil {
		return replayResult{}, err
	}
	result := replayResult{head: chain.head, sequence: chain.sequence, validEnd: storeHeaderBytes}
	return l.replayFrames(result, replayAnchor{}, apply)
}

func (l *recordLog) pendingRotation() (bool, error) {
	if len(l.sealed) == 0 {
		return false, nil
	}
	var header [frameHeaderBytes]byte
	if _, err := l.file.ReadAt(header[:], storeHeaderBytes); errors.Is(err, io.EOF) {
		return false, nil
	} else if err != nil {
		return false, err
	}
	_, record, err := decodeFrameHeader(header[:])
	if err != nil {
		return false, err
	}
	last := strings.TrimSuffix(filepath.Base(l.sealed[len(l.sealed)-1]), segmentExtension)
	sequence, err := strconv.ParseUint(last, segmentNameRadix, segmentNameBitSize)
	if err != nil {
		return false, err
	}
	return record.sequence <= sequence, nil
}

// finishRotation accepts only an exact duplicate of the validated final sealed
// segment. Readers leave bytes untouched; a writer completes the durable reset.
func (l *recordLog) finishRotation(chain replayResult) (replayResult, error) {
	sealed, err := os.Open(l.sealed[len(l.sealed)-1])
	if err != nil {
		return replayResult{}, err
	}
	defer sealed.Close()
	activeInfo, err := l.file.Stat()
	if err != nil {
		return replayResult{}, err
	}
	sealedInfo, err := sealed.Stat()
	if err != nil {
		return replayResult{}, err
	}
	if activeInfo.Size() != sealedInfo.Size() {
		return replayResult{}, errors.New("overgodb: interrupted rotation extent differs from sealed segment")
	}
	activeHash, sealedHash := sha256.New(), sha256.New()
	if _, err := io.Copy(activeHash, io.NewSectionReader(l.file, storeMagicOffset, activeInfo.Size())); err != nil {
		return replayResult{}, err
	}
	if _, err := io.Copy(sealedHash, sealed); err != nil {
		return replayResult{}, err
	}
	if !bytes.Equal(activeHash.Sum(nil), sealedHash.Sum(nil)) {
		return replayResult{}, errors.New("overgodb: interrupted rotation bytes differ from sealed segment")
	}
	end := activeInfo.Size()
	l.pendingReset = l.readOnly
	if !l.readOnly {
		if err := l.file.Truncate(storeHeaderBytes); err != nil {
			return replayResult{}, err
		}
		if err := l.file.Sync(); err != nil {
			return replayResult{}, err
		}
		end = storeHeaderBytes
	}
	return replayResult{head: chain.head, sequence: chain.sequence, validEnd: end}, nil
}

func (l *recordLog) replay(anchor replayAnchor, apply func(logRecord) error) (replayResult, error) {
	if anchor.sequence == 0 && anchor.head.Valid() || anchor.sequence != 0 && !anchor.head.Valid() {
		return replayResult{}, ErrSnapshotAnchor
	}
	if _, err := l.file.Seek(0, io.SeekStart); err != nil {
		return replayResult{}, fmt.Errorf("overgodb: seek log start: %w", err)
	}
	header := make([]byte, storeHeaderBytes)
	if _, err := io.ReadFull(l.file, header); err != nil {
		return replayResult{}, fmt.Errorf("overgodb: read log header: %w", err)
	}
	if err := validateStoreHeader(header); err != nil {
		return replayResult{}, err
	}
	if anchor.sequence != 0 {
		if anchor.offset < storeHeaderBytes {
			return replayResult{}, ErrSnapshotAnchor
		}
		digest, err := l.anchorDigest(anchor.offset)
		if err != nil || digest != anchor.digest {
			return replayResult{}, ErrSnapshotAnchor
		}
		if _, err := l.file.Seek(anchor.offset, io.SeekStart); err != nil {
			return replayResult{}, fmt.Errorf("overgodb: seek snapshot tail: %w", err)
		}
		return l.replayFrames(replayResult{
			head: anchor.head, sequence: anchor.sequence, validEnd: anchor.offset,
		}, anchor, apply)
	}
	result := replayResult{validEnd: storeHeaderBytes}
	return l.replayFrames(result, anchor, apply)
}

func (l *recordLog) anchorDigest(offset int64) ([sha256.Size]byte, error) {
	if offset < storeHeaderBytes {
		return [sha256.Size]byte{}, ErrSnapshotAnchor
	}
	start := max(int64(storeHeaderBytes), offset-sha256.Size)
	data := make([]byte, offset-start)
	if _, err := l.file.ReadAt(data, start); err != nil {
		return [sha256.Size]byte{}, err
	}
	return sha256.Sum256(data), nil
}

func (l *recordLog) refresh(anchor replayAnchor, offset int64, apply func(logRecord) error) (replayResult, error) {
	info, err := l.file.Stat()
	if err != nil {
		return replayResult{}, fmt.Errorf("overgodb: stat log tail: %w", err)
	}
	if info.Size() < offset {
		return replayResult{}, ErrSnapshotAnchor
	}
	if _, err := l.file.Seek(offset, io.SeekStart); err != nil {
		return replayResult{}, fmt.Errorf("overgodb: seek log tail: %w", err)
	}
	result := replayResult{head: anchor.head, sequence: anchor.sequence, validEnd: offset}
	return l.replayFrames(result, anchor, apply)
}

func (l *recordLog) replayFrames(result replayResult, anchor replayAnchor, apply func(logRecord) error) (replayResult, error) {
	frameHeader := make([]byte, frameHeaderBytes)
	var body []byte
	for {
		frameStart := result.validEnd
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
			return replayResult{}, fmt.Errorf("overgodb: frame at %d: %w", frameStart, err)
		}
		if payloadSize > maxFramePayload {
			return replayResult{}, fmt.Errorf("overgodb: frame at %d exceeds payload limit", frameStart)
		}
		bodyBytes := int(payloadSize) + crc32.Size
		if cap(body) < bodyBytes {
			body = make([]byte, bodyBytes)
		} else {
			body = body[:bodyBytes]
		}
		if _, err := io.ReadFull(l.file, body); err != nil {
			result.recovered = true
			break
		}
		record.payload = body[:payloadSize]
		record.segment = l.segment
		record.offset = frameStart + frameHeaderBytes
		checksum := binary.LittleEndian.Uint32(body[payloadSize:])
		actual := crc32.Checksum(frameHeader, crcTable)
		actual = crc32.Update(actual, crcTable, record.payload)
		if checksum != actual {
			return replayResult{}, fmt.Errorf("overgodb: frame at %d has invalid checksum", frameStart)
		}
		if record.sequence != result.sequence+1 || record.previous != result.head {
			// A mismatch on the first frame after an anchor is a wrong
			// anchor, not a broken journal: report it as such so the
			// caller falls back to full replay instead of refusing.
			if anchor.sequence != 0 && frameStart == anchor.offset {
				return replayResult{}, ErrSnapshotAnchor
			}
			return replayResult{}, fmt.Errorf("overgodb: frame at %d breaks commit chain", frameStart)
		}
		if record.id != commitIdentity(record.version, record.sequence, record.previous, record.payload) {
			return replayResult{}, fmt.Errorf("overgodb: frame at %d has invalid commit identity", frameStart)
		}
		if record.sequence == anchor.sequence && record.id != anchor.head {
			return replayResult{}, ErrSnapshotAnchor
		}
		if record.sequence > anchor.sequence {
			if err := apply(record); err != nil {
				return replayResult{}, fmt.Errorf("overgodb: apply frame at %d: %w", frameStart, err)
			}
		}
		result.sequence = record.sequence
		result.head = record.id
		result.validEnd += int64(frameHeaderBytes) + int64(payloadSize) + crc32.Size
	}
	if result.sequence < anchor.sequence {
		return replayResult{}, ErrSnapshotAnchor
	}
	return result, nil
}

func decodeFrameHeader(header []byte) (uint32, logRecord, error) {
	if binary.LittleEndian.Uint32(header[frameMagicOffset:frameVersionOffset]) != frameMagic {
		return 0, logRecord{}, errors.New("invalid frame magic")
	}
	version := binary.LittleEndian.Uint16(header[frameVersionOffset:frameKindOffset])
	if version != frameVersion {
		return 0, logRecord{}, fmt.Errorf(
			"unsupported frame version %d (this reader speaks %d); a pre-rearchitecture store must be migrated, not opened in place (docs/OVERGODB_IMPORT.md)",
			version, frameVersion)
	}
	if binary.LittleEndian.Uint16(header[frameKindOffset:frameSequenceOffset]) != frameKindBatch {
		return 0, logRecord{}, errors.New("unsupported frame kind")
	}
	record := logRecord{version: version, sequence: binary.LittleEndian.Uint64(header[frameSequenceOffset:framePreviousOffset])}
	copy(record.previous[:], header[framePreviousOffset:frameIDOffset])
	copy(record.id[:], header[frameIDOffset:framePayloadSizeOffset])
	return binary.LittleEndian.Uint32(header[framePayloadSizeOffset:frameHeaderBytes]), record, nil
}

func commitIdentity(version uint16, sequence uint64, previous artifact.CommitID, payload []byte) artifact.CommitID {
	hasher := sha256.New()
	var prefix [commitPrefixBytes]byte
	binary.LittleEndian.PutUint16(prefix[commitVersionOffset:commitKindOffset], version)
	binary.LittleEndian.PutUint16(prefix[commitKindOffset:commitSequenceOffset], frameKindBatch)
	binary.LittleEndian.PutUint64(prefix[commitSequenceOffset:commitPrefixBytes], sequence)
	_, _ = hasher.Write(prefix[:])
	_, _ = hasher.Write(previous[:])
	_, _ = hasher.Write(payload)
	var id artifact.CommitID
	copy(id[:], hasher.Sum(nil))
	return id
}

func encodeFrameHeader(version uint16, sequence uint64, previous, id artifact.CommitID, payloadSize int) []byte {
	header := make([]byte, frameHeaderBytes)
	binary.LittleEndian.PutUint32(header[frameMagicOffset:frameVersionOffset], frameMagic)
	binary.LittleEndian.PutUint16(header[frameVersionOffset:frameKindOffset], version)
	binary.LittleEndian.PutUint16(header[frameKindOffset:frameSequenceOffset], frameKindBatch)
	binary.LittleEndian.PutUint64(header[frameSequenceOffset:framePreviousOffset], sequence)
	copy(header[framePreviousOffset:frameIDOffset], previous[:])
	copy(header[frameIDOffset:framePayloadSizeOffset], id[:])
	binary.LittleEndian.PutUint32(header[framePayloadSizeOffset:frameHeaderBytes], uint32(payloadSize))
	return header
}

func encodeFrame(sequence uint64, previous artifact.CommitID, payload []byte) (artifact.CommitID, []byte, [crc32.Size]byte) {
	id := commitIdentity(frameVersion, sequence, previous, payload)
	header := encodeFrameHeader(frameVersion, sequence, previous, id, len(payload))
	checksum := crc32.Checksum(header, crcTable)
	checksum = crc32.Update(checksum, crcTable, payload)
	var trailer [crc32.Size]byte
	binary.LittleEndian.PutUint32(trailer[:], checksum)
	return id, header, trailer
}

func (l *recordLog) append(sequence uint64, previous artifact.CommitID, payload []byte) (artifact.CommitID, int64, int64, error) {
	if l == nil || l.file == nil || l.readOnly {
		return artifact.CommitID{}, 0, 0, errors.New("overgodb: log is not writable")
	}
	if len(payload) > maxFramePayload {
		return artifact.CommitID{}, 0, 0, errors.New("overgodb: batch exceeds payload limit")
	}
	info, err := l.file.Stat()
	if err != nil {
		return artifact.CommitID{}, 0, 0, fmt.Errorf("overgodb: locate append: %w", err)
	}
	id, header, trailer := encodeFrame(sequence, previous, payload)
	writer := l.writer
	if writer == nil {
		writer = l.file
	}
	payloadOffset := info.Size() + int64(len(header))
	end := payloadOffset + int64(len(payload)+len(trailer))
	for _, part := range [...][]byte{header, payload, trailer[:]} {
		if err := writeAll(writer, part); err != nil {
			return id, payloadOffset, end, fmt.Errorf("overgodb: append commit: %w", err)
		}
	}
	if err := writer.Sync(); err != nil {
		return id, payloadOffset, end, fmt.Errorf("overgodb: sync commit: %w", err)
	}
	return id, payloadOffset, end, nil
}

func (l *recordLog) openContent(locator contentLocator) io.Reader {
	return io.NewSectionReader(l.file, locator.offset, locator.size)
}

func writeAll(writer io.Writer, data []byte) error {
	for len(data) > 0 {
		written, err := writer.Write(data)
		if err != nil {
			return err
		}
		if written == 0 {
			return io.ErrShortWrite
		}
		data = data[written:]
	}
	return nil
}

// Close closes the underlying log file; further calls are no-ops.
func (l *recordLog) Close() error {
	if l == nil {
		return nil
	}
	var result error
	if l.file != nil {
		result = l.file.Close()
		l.file = nil
	}
	return result
}
