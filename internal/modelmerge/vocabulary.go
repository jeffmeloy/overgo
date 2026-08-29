package modelmerge

import (
	"cmp"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/tensor"
	"overgo/internal/tokenizer"
)

const (
	// EmbeddingProvenanceVersion is the immutable row-lineage document version.
	EmbeddingProvenanceVersion = artifact.InitialDocumentVersion
	// EmbeddingProvenanceMediaType identifies embedding row lineage.
	EmbeddingProvenanceMediaType = "application/vnd.overgo.embedding-provenance+json"
	// EmbeddingProvenanceSchema identifies the embedding provenance wire schema.
	EmbeddingProvenanceSchema = "overgo/embedding-provenance/v1"
)

// EmbeddingRowProvenance binds one generated embedding row to its exact source.
type EmbeddingRowProvenance struct {
	Tensor    string            `json:"tensor"`
	OutputRow tokenizer.TokenID `json:"output_row"`
	SourceRow tokenizer.TokenID `json:"source_row"`
}

// EmbeddingProvenance records every copied row in a vocabulary artifact.
type EmbeddingProvenance struct {
	Version          uint16                   `json:"version"`
	SourceModel      artifact.ID              `json:"source_model"`
	GeneratedModel   artifact.ID              `json:"generated_model"`
	SourceVocabulary artifact.ID              `json:"source_vocabulary"`
	TargetVocabulary artifact.ID              `json:"target_vocabulary"`
	Mapping          artifact.ID              `json:"mapping"`
	Rows             []EmbeddingRowProvenance `json:"rows"`
	ID               artifact.ID              `json:"-"`
}

// VocabularyArtifactResult carries the generated model and immutable row lineage.
type VocabularyArtifactResult struct {
	Model      Snapshot
	Mapping    tokenizer.VocabularyMapping
	Provenance EmbeddingProvenance
	Lineage    []artifact.Lineage
}

var embeddingProvenanceCodec = artifact.JSONDocumentCodec(
	"embedding provenance", artifact.KindEvidence,
	EmbeddingProvenanceMediaType, EmbeddingProvenanceSchema,
	canonicalizeEmbeddingProvenance,
	func(value EmbeddingProvenance) artifact.ID { return value.ID },
	func(value *EmbeddingProvenance, id artifact.ID) { value.ID = id },
	func(value EmbeddingProvenance) EmbeddingProvenance {
		value.Rows = slices.Clone(value.Rows)
		return value
	},
)

// VocabularyArtifact generates a model whose named embedding matrices follow
// the target vocabulary order. Every output row is copied from the exact source
// token; missing or semantically different tokens refuse.
func (compiler Compiler) VocabularyArtifact(
	source Snapshot,
	sourceVocabulary, targetVocabulary tokenizer.VocabularyArtifact,
	targetDefinition artifact.ID,
	embeddingTensors []string,
) (VocabularyArtifactResult, error) {
	if err := source.ValidateIdentity(); err != nil {
		return VocabularyArtifactResult{}, err
	}
	if targetDefinition.Kind() != artifact.KindModelDefinition {
		return VocabularyArtifactResult{}, errors.New("model merge: vocabulary target definition is invalid")
	}
	mapping, err := tokenizer.ExactVocabularyMapping(sourceVocabulary, targetVocabulary)
	if err != nil {
		return VocabularyArtifactResult{}, fmt.Errorf("model merge: compile vocabulary mapping: %w", err)
	}
	names, err := exactEmbeddingTensorNames(source, embeddingTensors)
	if err != nil {
		return VocabularyArtifactResult{}, err
	}
	weights := cloneWeights(source.Tensors)
	rows := make([]EmbeddingRowProvenance, 0, len(names)*len(mapping.Rows))
	for _, name := range names {
		remapped, remapErr := remapEmbeddingWeight(source.Tensors[name], mapping.Rows)
		if remapErr != nil {
			return VocabularyArtifactResult{}, fmt.Errorf("model merge: remap embedding %q: %w", name, remapErr)
		}
		weights[name] = remapped
		for outputRow, sourceRow := range mapping.Rows {
			rows = append(rows, EmbeddingRowProvenance{
				Tensor: name, OutputRow: tokenizer.TokenID(outputRow), SourceRow: sourceRow,
			})
		}
	}
	model, err := compiler.Seal(targetDefinition, source.ID, weights)
	if err != nil {
		return VocabularyArtifactResult{}, err
	}
	provenance, err := newEmbeddingProvenance(EmbeddingProvenance{
		Version: EmbeddingProvenanceVersion, SourceModel: source.ID, GeneratedModel: model.ID,
		SourceVocabulary: sourceVocabulary.ID, TargetVocabulary: targetVocabulary.ID,
		Mapping: mapping.ID, Rows: rows,
	})
	if err != nil {
		return VocabularyArtifactResult{}, err
	}
	lineage := artifact.DependencyLineage(
		model.ID, source.ID, sourceVocabulary.ID, targetVocabulary.ID, mapping.ID,
	)
	lineage = append(lineage, provenance.Lineage()...)
	result := VocabularyArtifactResult{
		Model: model, Mapping: mapping, Provenance: provenance, Lineage: lineage,
	}
	if err := result.ValidateGeneratedArtifact(source); err != nil {
		return VocabularyArtifactResult{}, err
	}
	return result, nil
}

// ValidateIdentity verifies immutable embedding provenance.
func (value EmbeddingProvenance) ValidateIdentity() error {
	return embeddingProvenanceCodec.ValidateIdentity(value)
}

// Content returns immutable embedding row provenance.
func (value EmbeddingProvenance) Content() (artifact.Content, error) {
	return embeddingProvenanceCodec.Content(value)
}

// Lineage binds provenance to the source, generated model, vocabularies, and mapping.
func (value EmbeddingProvenance) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(
		value.ID, value.SourceModel, value.GeneratedModel,
		value.SourceVocabulary, value.TargetVocabulary, value.Mapping,
	)
}

// Batch prepares atomic publication of embedding row provenance.
func (value EmbeddingProvenance) Batch(key string) (artifact.Batch, error) {
	return embeddingProvenanceCodec.Batch(key, value, value.Lineage(), nil)
}

// ValidateGeneratedArtifact verifies model identity, complete row provenance,
// and byte-equivalent F32 embedding rows against the named source model.
func (result VocabularyArtifactResult) ValidateGeneratedArtifact(source Snapshot) error {
	if err := source.ValidateIdentity(); err != nil {
		return err
	}
	if err := result.Model.ValidateIdentity(); err != nil {
		return err
	}
	if err := result.Mapping.ValidateIdentity(); err != nil {
		return err
	}
	if err := result.Provenance.ValidateIdentity(); err != nil {
		return err
	}
	if result.Model.Base != source.ID || result.Provenance.SourceModel != source.ID ||
		result.Provenance.GeneratedModel != result.Model.ID || result.Provenance.Mapping != result.Mapping.ID ||
		result.Provenance.SourceVocabulary != result.Mapping.Source ||
		result.Provenance.TargetVocabulary != result.Mapping.Target {
		return errors.New("model merge: generated vocabulary artifact authority differs")
	}
	byTensor := make(map[string][]EmbeddingRowProvenance)
	for _, row := range result.Provenance.Rows {
		byTensor[row.Tensor] = append(byTensor[row.Tensor], row)
	}
	if len(byTensor) == 0 {
		return errors.New("model merge: generated vocabulary artifact lacks embeddings")
	}
	for name, rows := range byTensor {
		before, sourceFound := source.Tensors[name]
		after, targetFound := result.Model.Tensors[name]
		if !sourceFound || !targetFound || len(rows) != len(result.Mapping.Rows) {
			return fmt.Errorf("model merge: generated embedding %q inventory differs", name)
		}
		width, sourceRows, sourceOK := tensor.MatrixExtents(before.Layout)
		targetWidth, targetRows, targetOK := tensor.MatrixExtents(after.Layout)
		widthInt, widthOK := checked.Int(width)
		if !sourceOK || !targetOK || !widthOK || targetWidth != width ||
			targetRows != uint64(len(result.Mapping.Rows)) {
			return fmt.Errorf("model merge: generated embedding %q geometry differs", name)
		}
		for _, row := range rows {
			output := int(row.OutputRow)
			sourceRow := int(row.SourceRow)
			if output < 0 || output >= len(result.Mapping.Rows) || sourceRow < 0 ||
				uint64(sourceRow) >= sourceRows || result.Mapping.Rows[output] != row.SourceRow {
				return fmt.Errorf("model merge: generated embedding %q row provenance differs", name)
			}
			outputStart, outputOK := checked.MulInt(output, widthInt)
			sourceStart, sourceOffsetOK := checked.MulInt(sourceRow, widthInt)
			outputEnd, outputEndOK := checked.AddInt(outputStart, widthInt)
			sourceEnd, sourceEndOK := checked.AddInt(sourceStart, widthInt)
			if !outputOK || !sourceOffsetOK || !outputEndOK || !sourceEndOK ||
				outputEnd > len(after.Values) || sourceEnd > len(before.Values) ||
				!slices.Equal(after.Values[outputStart:outputEnd], before.Values[sourceStart:sourceEnd]) {
				return fmt.Errorf("model merge: generated embedding %q row values differ", name)
			}
		}
	}
	return nil
}

func exactEmbeddingTensorNames(source Snapshot, names []string) ([]string, error) {
	if !checked.Nonempty(names) {
		return nil, errors.New("model merge: embedding tensor inventory is empty")
	}
	result := slices.Clone(names)
	slices.Sort(result)
	for index, name := range result {
		if name == "" || index > 0 && name == result[index-1] {
			return nil, errors.New("model merge: embedding tensor name is empty or duplicated")
		}
		if _, found := source.Tensors[name]; !found {
			return nil, fmt.Errorf("model merge: embedding tensor %q is absent", name)
		}
	}
	return result, nil
}

func remapEmbeddingWeight(weight Weight, rows []tokenizer.TokenID) (Weight, error) {
	width, sourceRows, valid := tensor.MatrixExtents(weight.Layout)
	widthInt, widthOK := checked.Int(width)
	targetValues, targetOK := checked.MulInt(widthInt, len(rows))
	if !valid || !widthOK || !targetOK {
		return Weight{}, errors.New("embedding is not a bounded matrix")
	}
	values := make([]float32, targetValues)
	for outputRow, sourceID := range rows {
		sourceRow := int(sourceID)
		if sourceRow < 0 || uint64(sourceRow) >= sourceRows {
			return Weight{}, errors.New("vocabulary row exceeds source embedding")
		}
		sourceStart, sourceOK := checked.MulInt(sourceRow, widthInt)
		outputStart, outputOK := checked.MulInt(outputRow, widthInt)
		sourceEnd, sourceEndOK := checked.AddInt(sourceStart, widthInt)
		outputEnd, outputEndOK := checked.AddInt(outputStart, widthInt)
		if !sourceOK || !outputOK || !sourceEndOK || !outputEndOK || sourceEnd > len(weight.Values) {
			return Weight{}, errors.New("embedding row storage differs from layout")
		}
		copy(values[outputStart:outputEnd], weight.Values[sourceStart:sourceEnd])
	}
	layout, err := tensor.NewShape(width, uint64(len(rows)))
	if err != nil {
		return Weight{}, err
	}
	return Weight{Layout: layout, Values: values}, nil
}

func newEmbeddingProvenance(value EmbeddingProvenance) (EmbeddingProvenance, error) {
	return embeddingProvenanceCodec.New(value)
}

func canonicalizeEmbeddingProvenance(value *EmbeddingProvenance) error {
	if value == nil || value.Version != EmbeddingProvenanceVersion ||
		value.SourceModel.Kind() != artifact.KindModel || value.GeneratedModel.Kind() != artifact.KindModel ||
		value.SourceVocabulary.Kind() != artifact.KindProfile || value.TargetVocabulary.Kind() != artifact.KindProfile ||
		value.Mapping.Kind() != artifact.KindProfile || !checked.Nonempty(value.Rows) {
		return errors.New("model merge: invalid embedding provenance")
	}
	slices.SortFunc(value.Rows, func(left, right EmbeddingRowProvenance) int {
		if order := cmp.Compare(left.Tensor, right.Tensor); order != 0 {
			return order
		}
		return cmp.Compare(left.OutputRow, right.OutputRow)
	})
	for index, row := range value.Rows {
		if row.Tensor == "" || row.OutputRow < 0 || row.SourceRow < 0 ||
			index > 0 && row.Tensor == value.Rows[index-1].Tensor && row.OutputRow == value.Rows[index-1].OutputRow {
			return errors.New("model merge: invalid or duplicated embedding provenance row")
		}
	}
	return nil
}
