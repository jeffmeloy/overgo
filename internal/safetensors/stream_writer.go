package safetensors

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
)

const (
	streamStagingSuffix                    = ".partial"
	streamArtifactName                     = "artifact"
	streamResumeName                       = "resume.json"
	streamSingleShardName                  = "model.safetensors"
	streamIndexName                        = "model.safetensors.index.json"
	streamIndexSizeKey                     = "total_size"
	privateStreamDirectoryMode os.FileMode = 0o700
	privateStreamFileMode      os.FileMode = 0o600
)

var ignoredExistingFileMode os.FileMode

type streamTensorSpec struct {
	Name  string   `json:"name"`
	DType string   `json:"dtype"`
	Shape []uint64 `json:"shape"`
}

type streamShardSpec struct {
	Name    string             `json:"name"`
	Tensors []streamTensorSpec `json:"tensors"`
}

type streamPlan struct {
	Shards   []streamShardSpec `json:"shards"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

type streamResume struct {
	Plan      string   `json:"plan"`
	Completed []string `json:"completed"`
}

type streamTensorLocation struct {
	shard  string
	offset int64
	bytes  uint64
}

type streamShardLayout struct {
	spec        streamShardSpec
	header      []byte
	payloadSize uint64
	fileSize    int64
}

type streamWriter struct {
	destination string
	staging     string
	plan        streamPlan
	planID      string
	locations   map[string]streamTensorLocation
	completed   map[string]struct{}
}

func newStreamWriter(destination string, plan streamPlan) (*streamWriter, error) {
	if strings.TrimSpace(destination) == "" {
		return nil, errors.New("safetensors: streaming destination is empty")
	}
	canonical, layouts, locations, planID, err := compileStreamPlan(plan)
	if err != nil {
		return nil, err
	}
	if _, err := os.Stat(destination); err == nil {
		return nil, errors.New("safetensors: streaming destination already exists")
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	writer := &streamWriter{
		destination: destination, staging: destination + streamStagingSuffix,
		plan: canonical, planID: planID, locations: locations,
		completed: make(map[string]struct{}),
	}
	if _, err := os.Stat(writer.staging); err == nil {
		if err := writer.resume(layouts); err != nil {
			return nil, err
		}
		return writer, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	if err := writer.initialize(layouts); err != nil {
		return nil, err
	}
	return writer, nil
}

func (writer *streamWriter) writeTensor(name string, payloadBytes uint64, payload io.Reader) error {
	if writer == nil || payload == nil {
		return errors.New("safetensors: streaming tensor source is absent")
	}
	location, found := writer.locations[name]
	if !found || payloadBytes != location.bytes {
		return fmt.Errorf("safetensors: streaming tensor %q extent differs", name)
	}
	if _, done := writer.completed[name]; done {
		return fmt.Errorf("safetensors: streaming tensor %q is already complete", name)
	}
	file, err := os.OpenFile(
		filepath.Join(writer.staging, streamArtifactName, location.shard),
		os.O_WRONLY,
		ignoredExistingFileMode,
	)
	if err != nil {
		return err
	}
	written, copyErr := io.CopyN(io.NewOffsetWriter(file, location.offset), payload, int64(payloadBytes))
	syncErr := file.Sync()
	closeErr := file.Close()
	if copyErr != nil || uint64(written) != payloadBytes || syncErr != nil || closeErr != nil {
		return errors.Join(errors.New("safetensors: streaming tensor write is incomplete"), copyErr, syncErr, closeErr)
	}
	writer.completed[name] = struct{}{}
	return writer.writeResume()
}

func (writer *streamWriter) finalize() error {
	if writer == nil || len(writer.completed) != len(writer.locations) {
		return errors.New("safetensors: streaming output is incomplete")
	}
	if len(writer.plan.Shards) > 1 {
		if err := writer.writeIndex(); err != nil {
			return err
		}
	}
	artifactDirectory := filepath.Join(writer.staging, streamArtifactName)
	if err := syncDirectory(artifactDirectory); err != nil {
		return err
	}
	if err := os.Rename(artifactDirectory, writer.destination); err != nil {
		return fmt.Errorf("safetensors: publish streaming output: %w", err)
	}
	if err := os.Remove(filepath.Join(writer.staging, streamResumeName)); err != nil {
		return err
	}
	if err := os.Remove(writer.staging); err != nil {
		return err
	}
	return syncDirectory(filepath.Dir(writer.destination))
}

func (writer *streamWriter) initialize(layouts []streamShardLayout) error {
	if err := os.Mkdir(writer.staging, privateStreamDirectoryMode); err != nil {
		return err
	}
	artifactDirectory := filepath.Join(writer.staging, streamArtifactName)
	if err := os.Mkdir(artifactDirectory, privateStreamDirectoryMode); err != nil {
		return err
	}
	for _, layout := range layouts {
		path := filepath.Join(artifactDirectory, layout.spec.Name)
		file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_RDWR, privateStreamFileMode)
		if err != nil {
			return err
		}
		if err := writeHeader(file, layout.header); err == nil {
			err = file.Truncate(layout.fileSize)
		}
		if err == nil {
			err = file.Sync()
		}
		closeErr := file.Close()
		if err != nil || closeErr != nil {
			return errors.Join(err, closeErr)
		}
	}
	return writer.writeResume()
}

func (writer *streamWriter) resume(layouts []streamShardLayout) error {
	data, err := os.ReadFile(filepath.Join(writer.staging, streamResumeName))
	if err != nil {
		return fmt.Errorf("safetensors: read streaming resume: %w", err)
	}
	var resume streamResume
	if err := json.Unmarshal(data, &resume); err != nil || resume.Plan != writer.planID {
		return errors.Join(errors.New("safetensors: streaming resume plan differs"), err)
	}
	for _, layout := range layouts {
		info, err := os.Stat(filepath.Join(writer.staging, streamArtifactName, layout.spec.Name))
		if err != nil || info.Size() != layout.fileSize {
			return errors.Join(errors.New("safetensors: streaming resume shard differs"), err)
		}
	}
	for _, name := range resume.Completed {
		if _, found := writer.locations[name]; !found {
			return errors.New("safetensors: streaming resume contains an unknown tensor")
		}
		if _, duplicate := writer.completed[name]; duplicate {
			return errors.New("safetensors: streaming resume contains a duplicate tensor")
		}
		writer.completed[name] = struct{}{}
	}
	return nil
}

func (writer *streamWriter) writeResume() error {
	names := make([]string, 0, len(writer.completed))
	for name := range writer.completed {
		names = append(names, name)
	}
	sort.Strings(names)
	data, err := json.Marshal(streamResume{Plan: writer.planID, Completed: names})
	if err != nil {
		return err
	}
	return replaceSyncedFile(filepath.Join(writer.staging, streamResumeName), data)
}

func (writer *streamWriter) writeIndex() error {
	weightMap := make(map[string]string, len(writer.locations))
	var total uint64
	for name, location := range writer.locations {
		weightMap[name] = location.shard
		var ok bool
		total, ok = addStreamBytes(total, location.bytes)
		if !ok {
			return errors.New("safetensors: streaming index extent overflows")
		}
	}
	index := struct {
		Metadata  map[string]uint64 `json:"metadata"`
		WeightMap map[string]string `json:"weight_map"`
	}{Metadata: map[string]uint64{streamIndexSizeKey: total}, WeightMap: weightMap}
	data, err := json.Marshal(index)
	if err != nil {
		return err
	}
	return replaceSyncedFile(filepath.Join(writer.staging, streamArtifactName, streamIndexName), data)
}

func compileStreamPlan(plan streamPlan) (streamPlan, []streamShardLayout, map[string]streamTensorLocation, string, error) {
	canonical := cloneStreamPlan(plan)
	sort.Slice(canonical.Shards, func(left, right int) bool { return canonical.Shards[left].Name < canonical.Shards[right].Name })
	locations := make(map[string]streamTensorLocation)
	layouts := make([]streamShardLayout, len(canonical.Shards))
	for shardIndex := range canonical.Shards {
		shard := &canonical.Shards[shardIndex]
		if filepath.Base(shard.Name) != shard.Name || !strings.HasSuffix(strings.ToLower(shard.Name), ".safetensors") || len(shard.Tensors) == 0 {
			return streamPlan{}, nil, nil, "", errors.New("safetensors: invalid streaming shard")
		}
		if shardIndex > 0 && canonical.Shards[shardIndex-1].Name == shard.Name {
			return streamPlan{}, nil, nil, "", errors.New("safetensors: duplicate streaming shard")
		}
		sort.Slice(shard.Tensors, func(left, right int) bool { return shard.Tensors[left].Name < shard.Tensors[right].Name })
		header := make(map[string]json.RawMessage, len(shard.Tensors)+len(canonical.Metadata))
		if len(canonical.Metadata) != 0 {
			metadata, err := json.Marshal(canonical.Metadata)
			if err != nil {
				return streamPlan{}, nil, nil, "", err
			}
			header["__metadata__"] = metadata
		}
		var payloadOffset uint64
		for tensorIndex, spec := range shard.Tensors {
			if spec.Name == "" || spec.DType == "" ||
				tensorIndex > 0 && shard.Tensors[tensorIndex-1].Name == spec.Name {
				return streamPlan{}, nil, nil, "", errors.New("safetensors: invalid streaming tensor specification")
			}
			if _, duplicate := locations[spec.Name]; duplicate {
				return streamPlan{}, nil, nil, "", errors.New("safetensors: duplicate streaming tensor")
			}
			bytes, err := TensorBytes(spec.DType, spec.Shape)
			if err != nil {
				return streamPlan{}, nil, nil, "", err
			}
			end, ok := addStreamBytes(payloadOffset, bytes)
			if !ok {
				return streamPlan{}, nil, nil, "", errors.New("safetensors: streaming shard extent overflows")
			}
			entry, err := json.Marshal(tensorHeader{DType: spec.DType, Shape: spec.Shape, DataOffsets: []uint64{payloadOffset, end}})
			if err != nil {
				return streamPlan{}, nil, nil, "", err
			}
			header[spec.Name] = entry
			if payloadOffset > math.MaxInt64 {
				return streamPlan{}, nil, nil, "", errors.New("safetensors: streaming tensor offset exceeds host limits")
			}
			locations[spec.Name] = streamTensorLocation{shard: shard.Name, offset: int64(payloadOffset), bytes: bytes}
			payloadOffset = end
		}
		headerBytes, err := paddedStreamHeader(header)
		if err != nil {
			return streamPlan{}, nil, nil, "", err
		}
		payloadStart := int64(len(headerBytes))
		if payloadStart > math.MaxInt64-int64(headerLengthPrefixBytes()) {
			return streamPlan{}, nil, nil, "", errors.New("safetensors: streaming header exceeds host offsets")
		}
		payloadStart += int64(headerLengthPrefixBytes())
		if payloadOffset > math.MaxInt64 || payloadStart > math.MaxInt64-int64(payloadOffset) {
			return streamPlan{}, nil, nil, "", errors.New("safetensors: streaming file exceeds host offsets")
		}
		for _, spec := range shard.Tensors {
			location := locations[spec.Name]
			location.offset += payloadStart
			locations[spec.Name] = location
		}
		layouts[shardIndex] = streamShardLayout{
			spec: *shard, header: headerBytes, payloadSize: payloadOffset,
			fileSize: payloadStart + int64(payloadOffset),
		}
	}
	if len(layouts) == 0 || len(locations) == 0 {
		return streamPlan{}, nil, nil, "", errors.New("safetensors: streaming plan is empty")
	}
	planBytes, err := json.Marshal(canonical)
	if err != nil {
		return streamPlan{}, nil, nil, "", err
	}
	digest := sha256.Sum256(planBytes)
	return canonical, layouts, locations, hex.EncodeToString(digest[:]), nil
}

func paddedStreamHeader(header map[string]json.RawMessage) ([]byte, error) {
	data, err := json.Marshal(header)
	if err != nil {
		return nil, err
	}
	return padHeaderBytes(data), nil
}

func replaceSyncedFile(path string, data []byte) error {
	temporary := path + streamStagingSuffix
	file, err := os.OpenFile(temporary, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, privateStreamFileMode)
	if err != nil {
		return err
	}
	if _, err = file.Write(data); err == nil {
		err = file.Sync()
	}
	closeErr := file.Close()
	if err != nil || closeErr != nil {
		return errors.Join(err, closeErr)
	}
	return os.Rename(temporary, path)
}

func cloneStreamPlan(plan streamPlan) streamPlan {
	metadata := plan.Metadata
	plan.Shards = slices.Clone(plan.Shards)
	for index := range plan.Shards {
		plan.Shards[index].Tensors = slices.Clone(plan.Shards[index].Tensors)
		for tensorIndex := range plan.Shards[index].Tensors {
			plan.Shards[index].Tensors[tensorIndex].Shape = slices.Clone(plan.Shards[index].Tensors[tensorIndex].Shape)
		}
	}
	plan.Metadata = make(map[string]string, len(metadata))
	for key, value := range metadata {
		plan.Metadata[key] = value
	}
	return plan
}

func addStreamBytes(left, right uint64) (uint64, bool) {
	if left > math.MaxUint64-right {
		var zero uint64
		return zero, false
	}
	return left + right, true
}
