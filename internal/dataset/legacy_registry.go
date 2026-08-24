package dataset

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
)

const (
	legacyDatasetRecord     byte = 23
	legacyDatasetFileRecord byte = 24
	legacyRecordKindBytes        = 1
	legacyRecordLengthBytes      = 4
	legacyIntegerBytes           = legacyRecordLengthBytes + legacyRecordLengthBytes
	legacyHeaderBytes            = legacyRecordKindBytes + legacyRecordLengthBytes
	legacyChecksumBytes          = legacyRecordLengthBytes
)

const (
	legacyBooleanFalse byte = iota
	legacyBooleanTrue
)

const legacyRecordKindOffset = iota

var legacyChecksumTable = crc32.MakeTable(crc32.Castagnoli)

type legacyDataset struct {
	ID        uint32
	Name      string
	Kind      string
	Source    string
	Modality  string
	FileCount int64
	ByteCount int64
}

type legacyDatasetFile struct {
	ID           uint32
	DatasetID    uint32
	Path         string
	OriginalName string
	Extension    string
	Modality     string
	Format       string
	Bytes        int64
	ModifiedUnix int64
	Structured   bool
	Attributes   string
}

type legacyRegistry struct {
	Datasets []legacyDataset
	Files    map[uint32][]legacyDatasetFile
}

func readLegacyRegistry(path string) (legacyRegistry, error) {
	file, err := os.Open(path)
	if err != nil {
		return legacyRegistry{}, err
	}
	defer file.Close()
	datasets := map[uint32]legacyDataset{}
	type fileKey struct {
		dataset uint32
		path    string
	}
	files := map[fileKey]legacyDatasetFile{}
	err = visitLegacyRecords(file, func(kind byte, payload []byte) error {
		switch kind {
		case legacyDatasetRecord:
			value, err := decodeLegacyDataset(payload)
			if err != nil {
				return err
			}
			datasets[value.ID] = value
		case legacyDatasetFileRecord:
			value, err := decodeLegacyDatasetFile(payload)
			if err != nil {
				return err
			}
			files[fileKey{dataset: value.DatasetID, path: value.Path}] = value
		default:
			return fmt.Errorf("dataset: unsupported legacy registry record %d", kind)
		}
		return nil
	})
	if err != nil {
		return legacyRegistry{}, err
	}
	result := legacyRegistry{Datasets: make([]legacyDataset, 0, len(datasets)), Files: map[uint32][]legacyDatasetFile{}}
	for _, value := range datasets {
		result.Datasets = append(result.Datasets, value)
	}
	sort.Slice(result.Datasets, func(i, j int) bool { return result.Datasets[i].Name < result.Datasets[j].Name })
	for _, value := range files {
		if _, ok := datasets[value.DatasetID]; !ok {
			return legacyRegistry{}, fmt.Errorf("dataset: file %q references absent legacy dataset %d", value.Path, value.DatasetID)
		}
		result.Files[value.DatasetID] = append(result.Files[value.DatasetID], value)
	}
	for id := range result.Files {
		slices.SortFunc(result.Files[id], func(left, right legacyDatasetFile) int {
			return strings.Compare(left.Path, right.Path)
		})
	}
	return result, nil
}

func visitLegacyRecords(reader io.Reader, visit func(byte, []byte) error) error {
	buffered := bufio.NewReader(reader)
	header := make([]byte, legacyHeaderBytes)
	for {
		if _, err := io.ReadFull(buffered, header); err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return fmt.Errorf("dataset: truncated legacy registry header: %w", err)
		}
		size := binary.LittleEndian.Uint32(header[1:])
		if size > artifact.MaxContentBytes {
			return errors.New("dataset: legacy registry record exceeds limit")
		}
		payload := make([]byte, int(size))
		if _, err := io.ReadFull(buffered, payload); err != nil {
			return fmt.Errorf("dataset: truncated legacy registry record: %w", err)
		}
		var checksum [legacyChecksumBytes]byte
		if _, err := io.ReadFull(buffered, checksum[:]); err != nil {
			return fmt.Errorf("dataset: truncated legacy registry checksum: %w", err)
		}
		want := binary.LittleEndian.Uint32(checksum[:])
		var got uint32
		got = crc32.Update(got, legacyChecksumTable, header)
		got = crc32.Update(got, legacyChecksumTable, payload)
		if got != want {
			return errors.New("dataset: legacy registry checksum mismatch")
		}
		if err := visit(header[legacyRecordKindOffset], payload); err != nil {
			return err
		}
	}
}

type legacyDecoder struct {
	data   []byte
	offset int
	err    error
}

func (decoder *legacyDecoder) uint32() uint32 {
	if decoder.err != nil || decoder.offset+legacyRecordLengthBytes > len(decoder.data) {
		decoder.err = errors.New("dataset: short legacy registry record")
		var zero uint32
		return zero
	}
	value := binary.LittleEndian.Uint32(decoder.data[decoder.offset:])
	decoder.offset += legacyRecordLengthBytes
	return value
}

func (decoder *legacyDecoder) int64() int64 {
	if decoder.err != nil || decoder.offset+legacyIntegerBytes > len(decoder.data) {
		decoder.err = errors.New("dataset: short legacy registry record")
		var zero int64
		return zero
	}
	value := int64(binary.LittleEndian.Uint64(decoder.data[decoder.offset:]))
	decoder.offset += legacyIntegerBytes
	return value
}

func (decoder *legacyDecoder) text() string {
	size := decoder.uint32()
	if decoder.err != nil || uint64(decoder.offset)+uint64(size) > uint64(len(decoder.data)) {
		decoder.err = errors.New("dataset: short legacy registry record")
		return ""
	}
	value := string(decoder.data[decoder.offset : decoder.offset+int(size)])
	decoder.offset += int(size)
	return value
}

func (decoder *legacyDecoder) boolean() bool {
	if decoder.err != nil || decoder.offset == len(decoder.data) {
		decoder.err = errors.New("dataset: short legacy registry record")
		return false
	}
	value := decoder.data[decoder.offset]
	decoder.offset++
	if value != legacyBooleanFalse && value != legacyBooleanTrue {
		decoder.err = errors.New("dataset: invalid legacy registry boolean")
	}
	return value == legacyBooleanTrue
}

func (decoder *legacyDecoder) finish() error {
	if decoder.err != nil {
		return decoder.err
	}
	if decoder.offset != len(decoder.data) {
		return errors.New("dataset: trailing legacy registry data")
	}
	return nil
}

func decodeLegacyDataset(payload []byte) (legacyDataset, error) {
	decoder := legacyDecoder{data: payload}
	value := legacyDataset{
		ID: decoder.uint32(), Name: decoder.text(), Kind: decoder.text(), Source: decoder.text(), Modality: decoder.text(),
		FileCount: decoder.int64(), ByteCount: decoder.int64(),
	}
	return value, decoder.finish()
}

func decodeLegacyDatasetFile(payload []byte) (legacyDatasetFile, error) {
	decoder := legacyDecoder{data: payload}
	value := legacyDatasetFile{
		ID: decoder.uint32(), DatasetID: decoder.uint32(), Path: decoder.text(), OriginalName: decoder.text(),
		Extension: decoder.text(), Modality: decoder.text(), Format: decoder.text(), Bytes: decoder.int64(),
		ModifiedUnix: decoder.int64(), Structured: decoder.boolean(), Attributes: decoder.text(),
	}
	return value, decoder.finish()
}
