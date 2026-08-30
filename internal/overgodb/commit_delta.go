package overgodb

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
)

type commitCoordinate struct {
	segment uint64
	offset  int64
	size    uint32
}

func (coordinate commitCoordinate) valid() bool {
	return coordinate.offset >= storeHeaderBytes && coordinate.size > 0 &&
		coordinate.size <= maxFramePayload
}

// CommitDelta is one validated durable journal transaction. Delta is the
// canonical persisted mutation with introduced content bytes materialized;
// Request is the canonical pre-reduction batch digest stored by that frame.
type CommitDelta struct {
	Commit   CommitView
	Previous artifact.CommitID
	Request  [sha256.Size]byte
	Delta    artifact.Batch
}

// CommitAt returns one exact commit coordinate from the validated commit
// projection without scanning the commit history.
func (s *Store) CommitAt(ctx context.Context, sequence uint64) (CommitView, bool, error) {
	if err := contextError(ctx); err != nil {
		return CommitView{}, false, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return CommitView{}, false, err
	}
	if sequence == 0 || sequence > uint64(s.state.commits.count()) {
		return CommitView{}, false, nil
	}
	commit := s.state.commits.at(int(sequence - 1))
	if commit.sequence != sequence {
		return CommitView{}, false, errors.New("overgodb: commit projection sequence differs")
	}
	return CommitView{Key: commit.key, ID: commit.id, Sequence: commit.sequence}, true, nil
}

// CommitDeltaAt reads exactly one indexed journal coordinate. It neither
// scans catalog contents nor replays preceding commits.
func (s *Store) CommitDeltaAt(ctx context.Context, sequence uint64) (CommitDelta, bool, error) {
	return s.commitDeltaAt(ctx, sequence, true)
}

// commitDeltaAt reads one indexed commit coordinate; materialize selects
// whether content payloads load. Metadata-only consumers — head-bound delta
// serving drops payload bytes anyway — skip the load entirely instead of
// copying content they discard.
func (s *Store) commitDeltaAt(ctx context.Context, sequence uint64, materialize bool) (CommitDelta, bool, error) {
	if err := contextError(ctx); err != nil {
		return CommitDelta{}, false, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return CommitDelta{}, false, err
	}
	if sequence == 0 || sequence > uint64(s.state.commits.count()) {
		return CommitDelta{}, false, nil
	}
	committed := s.state.commits.at(int(sequence - 1))
	if committed.sequence != sequence || !committed.coordinate.valid() {
		return CommitDelta{}, false, errors.New("overgodb: exact commit coordinate is unavailable")
	}
	reader := io.ReaderAt(s.log.file)
	var sealed *os.File
	if committed.coordinate.segment != activeSegment {
		path := filepath.Join(
			s.root,
			segmentDirectory,
			fmt.Sprintf("%020d%s", committed.coordinate.segment, segmentExtension),
		)
		var err error
		sealed, err = os.Open(path)
		if err != nil {
			return CommitDelta{}, false, fmt.Errorf("overgodb: open exact commit segment: %w", err)
		}
		defer sealed.Close()
		reader = sealed
	}
	record, err := readLogRecordAt(reader, committed.coordinate)
	if err != nil {
		return CommitDelta{}, false, fmt.Errorf("overgodb: read exact commit %d: %w", sequence, err)
	}
	record.segment = committed.coordinate.segment
	previous := artifact.CommitID{}
	if sequence > 1 {
		previous = s.state.commits.at(int(sequence - 2)).id
	}
	if record.sequence != sequence || record.id != committed.id || record.previous != previous {
		return CommitDelta{}, false, errors.New("overgodb: exact commit coordinate differs from catalog authority")
	}
	delta, request, _, err := decodeRecord(record)
	if err != nil {
		return CommitDelta{}, false, err
	}
	if delta.Key != committed.key || request != committed.payload {
		return CommitDelta{}, false, errors.New("overgodb: exact commit payload differs from catalog authority")
	}
	for index := range delta.Contents {
		if err := contextError(ctx); err != nil {
			return CommitDelta{}, false, err
		}
		id := delta.Contents[index].Descriptor.ID
		locator, found := s.state.contents.locator(id)
		if !found || locator.sequence != sequence {
			return CommitDelta{}, false, errors.New("overgodb: exact commit content introduction differs")
		}
		if !materialize {
			continue
		}
		data, err := s.materializeContent(id, locator)
		if err != nil {
			return CommitDelta{}, false, err
		}
		delta.Contents[index].Data = data
		if err := delta.Contents[index].Validate(); err != nil {
			return CommitDelta{}, false, err
		}
	}
	return CommitDelta{
		Commit:   CommitView{Key: committed.key, ID: committed.id, Sequence: committed.sequence},
		Previous: previous,
		Request:  request,
		Delta:    delta,
	}, true, nil
}

func readLogRecordAt(reader io.ReaderAt, coordinate commitCoordinate) (logRecord, error) {
	if reader == nil || !coordinate.valid() {
		return logRecord{}, errors.New("invalid commit coordinate")
	}
	header := make([]byte, frameHeaderBytes)
	if _, err := io.ReadFull(io.NewSectionReader(reader, coordinate.offset, frameHeaderBytes), header); err != nil {
		return logRecord{}, err
	}
	payloadSize, record, err := decodeFrameHeader(header)
	if err != nil {
		return logRecord{}, err
	}
	if payloadSize != coordinate.size {
		return logRecord{}, errors.New("commit coordinate payload size differs")
	}
	body := make([]byte, int(payloadSize)+crc32.Size)
	if _, err := io.ReadFull(
		io.NewSectionReader(reader, coordinate.offset+frameHeaderBytes, int64(len(body))),
		body,
	); err != nil {
		return logRecord{}, err
	}
	record.offset = coordinate.offset + frameHeaderBytes
	record.payload = body[:payloadSize]
	wantChecksum := binary.LittleEndian.Uint32(body[payloadSize:])
	checksum := crc32.Checksum(header, crcTable)
	checksum = crc32.Update(checksum, crcTable, record.payload)
	if checksum != wantChecksum {
		return logRecord{}, errors.New("commit coordinate checksum differs")
	}
	if record.id != commitIdentity(record.version, record.sequence, record.previous, record.payload) {
		return logRecord{}, errors.New("commit coordinate identity differs")
	}
	return record, nil
}
