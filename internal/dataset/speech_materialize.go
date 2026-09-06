package dataset

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"overgo/internal/artifact"
)

// SpeechRecord contains source coordinates and a target, never copied audio.
// Locations are repository facts and therefore cannot change record identity.
type SpeechRecord struct {
	Audio  artifact.ID        `json:"audio"`
	Origin AudioPayloadOrigin `json:"origin"`
	Target string             `json:"target"`
}

// SpeechMaterializationSpec binds a selection to an existing inventory version.
// Limit counts physical rows across inventory-ordered shards; zero selects all.
// Null audio and absent/empty transcripts are counted, not replaced by later rows.
type SpeechMaterializationSpec struct {
	Source       artifact.ID `json:"source"`
	AudioColumn  string      `json:"audio_column"`
	TargetColumn string      `json:"target_column"`
	Limit        uint64      `json:"limit,omitzero"`
}

// SpeechMaterialization reports reference-record publication, not audio decode
// or model execution. Rows = Records + Rejected for the visited selection.
type SpeechMaterialization struct {
	Dataset  artifact.ID `json:"dataset"`
	Profile  artifact.ID `json:"profile"`
	Rows     uint64      `json:"rows"`
	Records  uint64      `json:"records"`
	Rejected uint64      `json:"rejected"`
	Shards   uint64      `json:"shards"`
}

var speechRecordContract = artifact.DocumentContract{Kind: artifact.KindDatasetShard, MediaType: "application/x-ndjson", Schema: "overgo/speech-reference-records/v1"}

// MaterializeSpeechDataset derives training-ready reference records from an
// already registered directory inventory. Physical shards are hash-verified and
// referenced in place. The store owns only bounded JSONL metadata per shard;
// both external and inline metadata use trainingdata's ordinary record index.
// The caller supplies the inventory's root location and decode-workspace budget.
func MaterializeSpeechDataset(ctx context.Context, repository artifact.Repository, root string, spec SpeechMaterializationSpec, maximumBytes uint64) (SpeechMaterialization, error) {
	if ctx == nil || repository == nil || root == "" || spec.AudioColumn == "" || spec.TargetColumn == "" || spec.AudioColumn == spec.TargetColumn || maximumBytes == 0 {
		return SpeechMaterialization{}, errors.New("dataset: incomplete speech materialization")
	}
	source, found, err := Load(ctx, repository, spec.Source)
	if err != nil {
		return SpeechMaterialization{}, err
	}
	if !found || source.Type != TypeVersion || len(source.Assets) != 1 || source.Assets[0].Name != InventoryAssetName {
		return SpeechMaterialization{}, errors.New("dataset: speech source must be an existing inventory version")
	}
	inventory, found, err := loadInventory(ctx, repository, source.Assets[0].Artifact)
	if err != nil {
		return SpeechMaterialization{}, err
	}
	if !found {
		return SpeechMaterialization{}, errors.New("dataset: speech inventory is absent")
	}
	profile, err := artifact.JSONContent(artifact.JSONContract(artifact.KindProfile, "overgo/speech-materialization/v1"), spec)
	if err != nil {
		return SpeechMaterialization{}, err
	}
	result := SpeechMaterialization{Profile: profile.Descriptor.ID}
	batch := artifact.Batch{Key: "dataset/speech/" + result.Profile.String(), Contents: []artifact.Content{profile}, Lineage: []artifact.Lineage{{Child: result.Profile, Parent: spec.Source, Relation: artifact.RelationDerivedFrom}}}
	var assets []Asset
	for _, file := range inventory.Files {
		if spec.Limit != 0 && result.Rows >= spec.Limit {
			break
		}
		if err := ctx.Err(); err != nil {
			return result, err
		}
		if file.Format != "parquet" || !filepath.IsLocal(filepath.FromSlash(file.Path)) {
			return result, fmt.Errorf("dataset: unsupported speech inventory path %q", file.Path)
		}
		container, err := artifact.ParseID("dataset-shard:sha256:" + file.Digest)
		if err != nil {
			return result, err
		}
		path := filepath.Join(root, filepath.FromSlash(file.Path))
		rows, err := OpenParquetRows(ctx, path, []string{spec.AudioColumn, spec.TargetColumn}, maximumBytes)
		if err != nil {
			return result, err
		}
		content, visited, rejected, err := materializeSpeechShard(ctx, rows, container, file.Bytes, spec, result.Rows, maximumBytes)
		closeErr := rows.Close()
		if err != nil || closeErr != nil {
			return result, errors.Join(err, closeErr)
		}
		result.Rows += visited
		result.Rejected += rejected
		result.Shards++
		if visited == rejected {
			continue
		}
		result.Records += visited - rejected
		assets = append(assets, Asset{Name: file.Path, Artifact: content.Descriptor.ID, Records: visited - rejected})
		batch.Contents = append(batch.Contents, content)
		descriptor, found, err := repository.Artifact(ctx, container)
		if err != nil {
			return result, err
		}
		if found && descriptor.Size != file.Bytes {
			return result, errors.New("dataset: registered speech shard size differs")
		}
		if !found {
			descriptor = artifact.Descriptor{ID: container, Size: file.Bytes, MediaType: "application/vnd.apache.parquet"}
		}
		batch.Artifacts = append(batch.Artifacts, descriptor)
		batch.Lineage = append(batch.Lineage, artifact.Lineage{Child: content.Descriptor.ID, Parent: container, Relation: artifact.RelationDerivedFrom}, artifact.Lineage{Child: content.Descriptor.ID, Parent: result.Profile, Relation: artifact.RelationDependsOn})
		location, err := artifact.CanonicalLocalLocation(container, artifact.LocationFile, path)
		if err != nil {
			return result, err
		}
		batch.Locations = append(batch.Locations, artifact.LocationEvent{Location: location, Action: artifact.LocationAdd})
	}
	if result.Records == 0 || spec.Limit != 0 && result.Rows != spec.Limit {
		return result, errors.New("dataset: speech selection is empty or shorter than declared")
	}
	version, err := NewVersion(assets)
	if err != nil {
		return result, err
	}
	result.Dataset = version.ID
	content, err := version.Content()
	if err != nil {
		return result, err
	}
	batch.Contents = append(batch.Contents, content)
	batch.Lineage = append(batch.Lineage, version.Lineage()...)
	batch.Lineage = append(batch.Lineage, artifact.Lineage{Child: version.ID, Parent: spec.Source, Relation: artifact.RelationDerivedFrom})
	report, err := artifact.JSONContent(artifact.JSONContract(artifact.KindEvidence, "overgo/speech-materialization-report/v1"), result)
	if err != nil {
		return result, err
	}
	batch.Contents = append(batch.Contents, report)
	batch.Lineage = append(batch.Lineage, artifact.Lineage{Child: report.Descriptor.ID, Parent: version.ID, Relation: artifact.RelationDerivedFrom}, artifact.Lineage{Child: report.Descriptor.ID, Parent: result.Profile, Relation: artifact.RelationDependsOn})
	_, err = artifact.CommitBatch(ctx, repository, batch)
	return result, err
}

func materializeSpeechShard(ctx context.Context, rows *ParquetRows, container artifact.ID, expectedBytes uint64, spec SpeechMaterializationSpec, already, maximumBytes uint64) (artifact.Content, uint64, uint64, error) {
	id, count, err := artifact.Identify(container.Kind(), audioReadFunc(func(p []byte) (int, error) {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		return rows.file.Read(p)
	}))
	if err != nil || id != container || count != expectedBytes {
		return artifact.Content{}, 0, 0, errors.Join(errors.New("dataset: speech shard identity or size differs"), err)
	}
	limit := rows.Rows()
	if spec.Limit != 0 {
		limit = min(limit, spec.Limit-already)
	}
	var output bytes.Buffer
	var rejected uint64
	for row := uint64(0); row < limit; row++ {
		values, err := rows.Read(ctx, row)
		if err != nil {
			return artifact.Content{}, row, rejected, err
		}
		audio, target := values[spec.AudioColumn], values[spec.TargetColumn]
		if audio == nil || len(*audio) == 0 || target == nil || !utf8.ValidString(*target) || strings.TrimSpace(*target) == "" {
			rejected++
			continue
		}
		id, err := artifact.IdentifyBytes(artifact.KindFile, []byte(*audio))
		if err != nil {
			return artifact.Content{}, row, rejected, err
		}
		record := SpeechRecord{Audio: id, Origin: AudioPayloadOrigin{Container: container, Column: spec.AudioColumn, Row: new(row)}, Target: *target}
		line, err := json.Marshal(record)
		if err != nil {
			return artifact.Content{}, row, rejected, err
		}
		bound := min(maximumBytes, uint64(artifact.MaxContentBytes))
		if uint64(len(line))+1 > bound-uint64(output.Len()) {
			return artifact.Content{}, row, rejected, errors.New("dataset: speech reference metadata exceeds shard budget")
		}
		_, _ = output.Write(line)
		_ = output.WriteByte('\n')
	}
	if output.Len() == 0 {
		return artifact.Content{}, limit, rejected, nil
	}
	content, err := speechRecordContract.OwnedContentBytes(output.Bytes())
	return content, limit, rejected, err
}
