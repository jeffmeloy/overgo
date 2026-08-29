package dataset

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"sort"
	"strconv"
	"strings"

	"overgo/internal/artifact"
)

const (
	BenchmarkFormatJSONL     = "jsonl"
	BenchmarkFormatJSONArray = "json-array"
	// BenchmarkFormatArrow decodes an Arrow IPC stream, the layout the
	// HuggingFace dataset cache holds lm_eval benchmarks in.
	BenchmarkFormatArrow = "arrow"
	// BenchmarkFormatParquetText streams one string column of a parquet
	// corpus file, bounded by the spec's limit -- the shape corpus-slice
	// evaluations import. The single field binding names the column.
	BenchmarkFormatParquetText = "parquet-text"
	// BenchmarkColumnAutoText asks the parquet reader to pick the corpus
	// text column by its conventional names (sequence, then text).
	BenchmarkColumnAutoText  = "auto-text"
	benchmarkRecordMediaType = "application/vnd.overgo.benchmark-record+json"
	benchmarkRecordSchema    = "overgo/benchmark-record/v1"
	benchmarkImportMediaType = "application/vnd.overgo.benchmark-import+json"
	benchmarkImportSchema    = "overgo/benchmark-import/v1"
	benchmarkImportVersion   = 1
)

var benchmarkRecordCodec = artifact.JSONDocumentCodec(
	"benchmark record", artifact.KindDatasetShard, benchmarkRecordMediaType, benchmarkRecordSchema,
	canonicalizeBenchmarkRecord, func(value BenchmarkRecord) artifact.ID { return value.ID },
	func(value *BenchmarkRecord, id artifact.ID) { value.ID = id }, cloneBenchmarkRecord,
)

var benchmarkImportCodec = artifact.JSONDocumentCodec(
	"benchmark import", artifact.KindDataset, benchmarkImportMediaType, benchmarkImportSchema,
	canonicalizeBenchmarkImport, func(value BenchmarkImport) artifact.ID { return value.ID },
	func(value *BenchmarkImport, id artifact.ID) { value.ID = id }, cloneBenchmarkImport,
)

type FieldBinding struct {
	Target string `json:"target"`
	Source string `json:"source"`
}

type BenchmarkImportSpec struct {
	Source     string         `json:"source"`
	Revision   string         `json:"revision"`
	SHA256     string         `json:"sha256"`
	Split      string         `json:"split"`
	Format     string         `json:"format"`
	Conversion string         `json:"conversion"`
	Fields     []FieldBinding `json:"fields"`
	// Limit bounds a corpus-slice import (parquet-text only): the spec
	// declares exactly how many rows the slice holds, so the import is
	// reproducible against the same bytes.
	Limit uint64 `json:"limit,omitzero"`
}

type BenchmarkField struct {
	Name  string          `json:"name"`
	Value json.RawMessage `json:"value"`
}

type BenchmarkRecord struct {
	ID      artifact.ID      `json:"-"`
	Profile artifact.ID      `json:"profile"`
	Ordinal uint64           `json:"ordinal"`
	Fields  []BenchmarkField `json:"fields"`
}

type BenchmarkImport struct {
	ID      artifact.ID         `json:"-"`
	Version uint16              `json:"version"`
	Spec    BenchmarkImportSpec `json:"spec"`
	Profile artifact.ID         `json:"profile"`
	Records []artifact.ID       `json:"records"`
	Count   uint64              `json:"count"`
}

func ImportBenchmark(ctx context.Context, repository artifact.Repository, path string, spec BenchmarkImportSpec) (BenchmarkImport, error) {
	if ctx == nil || repository == nil {
		return BenchmarkImport{}, errors.New("dataset: benchmark repository is absent")
	}
	canonical, profile, err := compileBenchmarkImport(spec)
	if err != nil {
		return BenchmarkImport{}, err
	}
	observed, err := HashFile(path)
	if err != nil {
		return BenchmarkImport{}, err
	}
	if observed != canonical.SHA256 {
		return BenchmarkImport{}, errors.New("dataset: benchmark source hash differs")
	}
	result := BenchmarkImport{Version: benchmarkImportVersion, Spec: canonical, Profile: profile}
	observe := func(ordinal uint64, source map[string]json.RawMessage) error {
		fields := make([]BenchmarkField, len(canonical.Fields))
		for index, binding := range canonical.Fields {
			value, ok := source[binding.Source]
			if !ok {
				return fmt.Errorf("dataset: benchmark field %q is absent", binding.Source)
			}
			fields[index] = BenchmarkField{Name: binding.Target, Value: slices.Clone(value)}
		}
		record, err := benchmarkRecordCodec.New(BenchmarkRecord{Profile: profile, Ordinal: ordinal, Fields: fields})
		if err != nil {
			return err
		}
		batch, err := benchmarkRecordCodec.Batch(
			"dataset/benchmark/record/"+profile.String()+"/"+strconv.FormatUint(ordinal, 10), record, nil, nil,
		)
		if err != nil {
			return err
		}
		if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
			return err
		}
		result.Records = append(result.Records, record.ID)
		return nil
	}
	if canonical.Format == BenchmarkFormatParquetText {
		column := canonical.Fields[0].Source
		if column == BenchmarkColumnAutoText {
			column = ""
		}
		err = ReadParquetTextRows(path, column, int(canonical.Limit), func(ordinal uint64, text string) error {
			encoded, err := json.Marshal(text)
			if err != nil {
				return err
			}
			return observe(ordinal, map[string]json.RawMessage{canonical.Fields[0].Source: encoded})
		})
	} else {
		file, openErr := os.Open(path)
		if openErr != nil {
			return BenchmarkImport{}, openErr
		}
		defer file.Close()
		err = decodeBenchmark(file, canonical.Format, observe)
	}
	if err != nil {
		return BenchmarkImport{}, err
	}
	result.Count = uint64(len(result.Records))
	result, err = benchmarkImportCodec.New(result)
	if err != nil {
		return BenchmarkImport{}, err
	}
	batch, err := benchmarkImportCodec.Batch(
		"dataset/benchmark/import/"+result.ID.String(), result,
		artifact.DependencyLineage(result.ID, result.Records...), nil,
	)
	if err != nil {
		return BenchmarkImport{}, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return BenchmarkImport{}, err
	}
	return result, nil
}

// ReadBenchmarkImport loads one committed benchmark import document.
func ReadBenchmarkImport(ctx context.Context, reader artifact.Reader, id artifact.ID) (BenchmarkImport, bool, error) {
	return benchmarkImportCodec.Read(ctx, reader, id)
}

// ReadBenchmarkRecord loads one committed benchmark case record.
func ReadBenchmarkRecord(ctx context.Context, reader artifact.Reader, id artifact.ID) (BenchmarkRecord, bool, error) {
	return benchmarkRecordCodec.Read(ctx, reader, id)
}

func compileBenchmarkImport(spec BenchmarkImportSpec) (BenchmarkImportSpec, artifact.ID, error) {
	spec.Source = strings.TrimSpace(spec.Source)
	spec.Revision = strings.TrimSpace(spec.Revision)
	spec.SHA256 = strings.ToLower(strings.TrimSpace(spec.SHA256))
	spec.Split = strings.TrimSpace(spec.Split)
	spec.Conversion = strings.TrimSpace(spec.Conversion)
	if spec.Source == "" || spec.Revision == "" || spec.Split == "" || spec.Conversion == "" || len(spec.Fields) == 0 ||
		(spec.Format != BenchmarkFormatJSONL && spec.Format != BenchmarkFormatJSONArray &&
			spec.Format != BenchmarkFormatArrow && spec.Format != BenchmarkFormatParquetText) {
		return BenchmarkImportSpec{}, artifact.ID{}, errors.New("dataset: incomplete benchmark import")
	}
	if (spec.Format == BenchmarkFormatParquetText) != (spec.Limit > 0) {
		return BenchmarkImportSpec{}, artifact.ID{}, errors.New("dataset: the slice limit belongs to parquet-text imports exactly")
	}
	if spec.Format == BenchmarkFormatParquetText && len(spec.Fields) != 1 {
		return BenchmarkImportSpec{}, artifact.ID{}, errors.New("dataset: parquet-text imports bind exactly one column")
	}
	digest, err := hex.DecodeString(spec.SHA256)
	if err != nil || len(digest) != sha256.Size {
		return BenchmarkImportSpec{}, artifact.ID{}, errors.New("dataset: invalid benchmark source hash")
	}
	spec.Fields = slices.Clone(spec.Fields)
	sort.Slice(spec.Fields, func(i, j int) bool { return spec.Fields[i].Target < spec.Fields[j].Target })
	for index, binding := range spec.Fields {
		if strings.TrimSpace(binding.Target) != binding.Target || binding.Target == "" ||
			strings.TrimSpace(binding.Source) != binding.Source || binding.Source == "" ||
			index > 0 && spec.Fields[index-1].Target == binding.Target {
			return BenchmarkImportSpec{}, artifact.ID{}, errors.New("dataset: invalid benchmark field mapping")
		}
	}
	profile, err := artifact.JSONID(artifact.KindProfile, spec)
	return spec, profile, err
}

func canonicalizeBenchmarkRecord(record *BenchmarkRecord) error {
	if record == nil || record.Profile.Kind() != artifact.KindProfile || len(record.Fields) == 0 {
		return errors.New("dataset: invalid benchmark record")
	}
	for index, field := range record.Fields {
		if strings.TrimSpace(field.Name) != field.Name || field.Name == "" || !json.Valid(field.Value) ||
			index > 0 && record.Fields[index-1].Name >= field.Name {
			return errors.New("dataset: invalid benchmark record field")
		}
	}
	return nil
}

func canonicalizeBenchmarkImport(value *BenchmarkImport) error {
	if value == nil || value.Version != benchmarkImportVersion || value.Profile.Kind() != artifact.KindProfile ||
		value.Count == 0 || value.Count != uint64(len(value.Records)) {
		return errors.New("dataset: invalid benchmark import")
	}
	spec, profile, err := compileBenchmarkImport(value.Spec)
	if err != nil || profile != value.Profile {
		return errors.New("dataset: benchmark import profile differs")
	}
	value.Spec = spec
	for _, record := range value.Records {
		if record.Kind() != artifact.KindDatasetShard {
			return errors.New("dataset: invalid benchmark record identity")
		}
	}
	return nil
}

func cloneBenchmarkRecord(record BenchmarkRecord) BenchmarkRecord {
	record.Fields = slices.Clone(record.Fields)
	for index := range record.Fields {
		record.Fields[index].Value = slices.Clone(record.Fields[index].Value)
	}
	return record
}

func cloneBenchmarkImport(value BenchmarkImport) BenchmarkImport {
	value.Spec.Fields = slices.Clone(value.Spec.Fields)
	value.Records = slices.Clone(value.Records)
	return value
}

// HashFile digests one file with the same claim the import records:
// the SHA-256 of these bytes as read. Cache scanners derive their
// import specs from it so a moved or edited file refuses to import.
func HashFile(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func decodeBenchmark(reader io.Reader, format string, observe func(uint64, map[string]json.RawMessage) error) error {
	if format == BenchmarkFormatArrow {
		return DecodeArrowStream(reader, observe)
	}
	decoder := json.NewDecoder(reader)
	if format == BenchmarkFormatJSONL {
		return decodeBenchmarkValues(decoder, observe)
	}
	token, err := decoder.Token()
	if err != nil || token != json.Delim('[') {
		return errors.New("dataset: benchmark JSON array is invalid")
	}
	var ordinal uint64
	for decoder.More() {
		var value map[string]json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return err
		}
		if err := observe(ordinal, value); err != nil {
			return err
		}
		ordinal++
	}
	token, err = decoder.Token()
	if err != nil || token != json.Delim(']') {
		return errors.New("dataset: benchmark JSON array is incomplete")
	}
	return nil
}

func decodeBenchmarkValues(decoder *json.Decoder, observe func(uint64, map[string]json.RawMessage) error) error {
	for ordinal := uint64(0); ; ordinal++ {
		var value map[string]json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}
		if err := observe(ordinal, value); err != nil {
			return err
		}
	}
}
