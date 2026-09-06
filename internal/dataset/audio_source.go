package dataset

import (
	"context"
	"errors"
	"io"
	"os"
	"sync"

	"overgo/internal/artifact"
)

// AudioPayloadReference contains execution input only: a location, container
// coordinate and expected encoded-payload identity. Transcripts belong in the
// training target or evaluation suite, never this reference.
type AudioPayloadReference struct {
	Path   string             `json:"path"`
	Audio  artifact.ID        `json:"audio"`
	Origin AudioPayloadOrigin `json:"origin"`
}

// AudioPayloadReader shares one bounded Parquet group between training and
// evaluation reads. Whole-file audio is read on demand without a corpus cache.
// The caller owns Close and must not modify source files during a pass.
type AudioPayloadReader struct {
	mu           sync.Mutex
	maximumBytes uint64
	path, column string
	container    artifact.ID
	rows         *ParquetRows
	stamp        os.FileInfo
	closed       bool
}

// NewAudioPayloadReader requires an explicit container decode-workspace budget.
func NewAudioPayloadReader(maximumBytes uint64) (*AudioPayloadReader, error) {
	if maximumBytes == 0 || maximumBytes > uint64(^uint(0)>>1) {
		return nil, errors.New("dataset: invalid audio source workspace")
	}
	return &AudioPayloadReader{maximumBytes: maximumBytes}, nil
}

// Read resolves and verifies one payload without accepting any target text.
// maximumEncodedBytes bounds the returned byte slice separately from Parquet
// workspace. Identity mismatch, missing rows and null payloads are errors.
func (reader *AudioPayloadReader) Read(ctx context.Context, reference AudioPayloadReference, maximumEncodedBytes uint64) ([]byte, error) {
	if ctx == nil || reference.Path == "" || reference.Audio.Kind() != artifact.KindFile || maximumEncodedBytes == 0 || maximumEncodedBytes > uint64(^uint(0)>>1) {
		return nil, errors.New("dataset: incomplete audio payload reference")
	}
	if err := reference.Origin.Validate(); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	reader.mu.Lock()
	defer reader.mu.Unlock()
	if reader.closed {
		return nil, os.ErrClosed
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	var data []byte
	if reference.Origin.Column == "" {
		file, err := os.Open(reference.Path)
		if err != nil {
			return nil, err
		}
		defer file.Close()
		info, err := file.Stat()
		if err != nil {
			return nil, err
		}
		if !info.Mode().IsRegular() || info.Size() <= 0 || uint64(info.Size()) > maximumEncodedBytes {
			return nil, errors.New("dataset: audio file exceeds payload bound or is empty")
		}
		data = make([]byte, int(info.Size()))
		if _, err := io.ReadFull(audioReadFunc(func(p []byte) (int, error) {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
			return file.Read(p)
		}), data); err != nil {
			return nil, err
		}
		var excess [1]byte
		if n, err := file.Read(excess[:]); n != 0 || !errors.Is(err, io.EOF) {
			return nil, errors.New("dataset: audio file changed during read")
		}
		container, err := artifact.IdentifyBytes(reference.Origin.Container.Kind(), data)
		if err != nil || container != reference.Origin.Container {
			return nil, errors.New("dataset: audio container identity differs")
		}
	} else {
		if reader.rows == nil || reader.path != reference.Path || reader.column != reference.Origin.Column || reader.container != reference.Origin.Container {
			if reader.rows != nil {
				_ = reader.rows.Close()
				reader.rows = nil
			}
			rows, err := OpenParquetRows(ctx, reference.Path, []string{reference.Origin.Column}, reader.maximumBytes)
			if err != nil {
				return nil, err
			}
			id, _, err := artifact.Identify(reference.Origin.Container.Kind(), audioReadFunc(func(p []byte) (int, error) {
				if err := ctx.Err(); err != nil {
					return 0, err
				}
				return rows.file.Read(p)
			}))
			if err != nil || id != reference.Origin.Container {
				_ = rows.Close()
				return nil, errors.Join(errors.New("dataset: parquet container identity differs"), err)
			}
			stamp, err := rows.file.Stat()
			if err != nil {
				_ = rows.Close()
				return nil, err
			}
			reader.rows, reader.path, reader.column, reader.container, reader.stamp = rows, reference.Path, reference.Origin.Column, reference.Origin.Container, stamp
		}
		stamp, err := reader.rows.file.Stat()
		if err != nil {
			return nil, err
		}
		if stamp.Size() != reader.stamp.Size() || !stamp.ModTime().Equal(reader.stamp.ModTime()) {
			return nil, errors.New("dataset: parquet source changed during pass")
		}
		ordinal := reference.Origin.ValueIndex
		if reference.Origin.Row != nil {
			ordinal = *reference.Origin.Row
		}
		var value *string
		if reference.Origin.Row == nil {
			// ValueIndex is an immutable legacy coordinate. Resolve it through
			// the same bounded reader, without treating it as a physical row.
			var present uint64
			for row := uint64(0); row < reader.rows.Rows(); row++ {
				values, err := reader.rows.Read(ctx, row)
				if err != nil {
					return nil, err
				}
				if candidate := values[reference.Origin.Column]; candidate != nil {
					if present == ordinal {
						value = candidate
						break
					}
					present++
				}
			}
		} else {
			values, err := reader.rows.Read(ctx, ordinal)
			if err != nil {
				return nil, err
			}
			value = values[reference.Origin.Column]
		}
		if value == nil || len(*value) == 0 || uint64(len(*value)) > maximumEncodedBytes {
			return nil, errors.New("dataset: audio row is absent, null, empty or exceeds payload bound")
		}
		data = []byte(*value)
	}
	id, err := artifact.IdentifyBytes(artifact.KindFile, data)
	if err != nil || id != reference.Audio {
		return nil, errors.New("dataset: encoded audio identity differs")
	}
	return data, nil
}

// Close releases the reader's cached source and refuses subsequent reads.
func (reader *AudioPayloadReader) Close() error {
	reader.mu.Lock()
	defer reader.mu.Unlock()
	reader.closed = true
	if reader.rows == nil {
		return nil
	}
	err := reader.rows.Close()
	reader.rows = nil
	return err
}

type audioReadFunc func([]byte) (int, error)

// Read delegates one context-aware source read.
func (read audioReadFunc) Read(p []byte) (int, error) { return read(p) }
