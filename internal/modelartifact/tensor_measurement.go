package modelartifact

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
	"overgo/internal/tensorstats"
)

const (
	TensorMeasurementVersion   uint16 = 2
	TensorMeasurementMediaType        = "application/vnd.overgo.tensor-measurement+json"
	TensorMeasurementSchema           = "overgo/tensor-measurement/v2"
)

var tensorMeasurementContract = artifact.DocumentContract{
	Kind: artifact.KindTensorInventory, MediaType: TensorMeasurementMediaType, Schema: TensorMeasurementSchema,
}

var tensorMeasurementCodec = artifact.DocumentCodec[TensorMeasurementDocument]{
	Name: "model artifact tensor measurement", Contract: tensorMeasurementContract,
	Decode: func(data []byte, value *TensorMeasurementDocument) error {
		return strictjson.DecodeBytes(data, value)
	},
	Encode: tensorMeasurementContent, Canonicalize: canonicalizeTensorMeasurements,
	Clone: func(value TensorMeasurementDocument) TensorMeasurementDocument {
		value.Measurements = slices.Clone(value.Measurements)
		return value
	},
	Identity:    func(value TensorMeasurementDocument) artifact.ID { return value.ID },
	SetIdentity: func(value *TensorMeasurementDocument, id artifact.ID) { value.ID = id },
}

type MeasurementPolicy struct {
	MaxSamplesPerTensor uint64 `json:"max_samples_per_tensor"`
	MaxReadBytes        uint64 `json:"max_read_bytes"`
}

// TensorMeasurement is one tensor's distribution-free characterization: robust
// L-moments, order statistics, and energy descriptors over a bounded sample.
// The statistics come from the shared tensorstats package; only Name is local.
type TensorMeasurement struct {
	Name string `json:"name"`
	tensorstats.Characterization
}

// TensorMeasurementDocument: bounded sampled tensor evidence.
type TensorMeasurementDocument struct {
	Version      uint16              `json:"version"`
	Inventory    artifact.ID         `json:"inventory"`
	Policy       MeasurementPolicy   `json:"policy"`
	ReadBytes    uint64              `json:"read_bytes"`
	Measurements []TensorMeasurement `json:"measurements"`
	ID           artifact.ID         `json:"-"`
}

func newTensorMeasurementDocument(
	inventory artifact.ID,
	policy MeasurementPolicy,
	readBytes uint64,
	measurements []TensorMeasurement,
) (TensorMeasurementDocument, error) {
	document := TensorMeasurementDocument{
		Version: TensorMeasurementVersion, Inventory: inventory, Policy: policy,
		ReadBytes: readBytes, Measurements: slices.Clone(measurements),
	}
	return tensorMeasurementCodec.New(document)
}

func (d TensorMeasurementDocument) ValidateIdentity() error {
	return tensorMeasurementCodec.ValidateIdentity(d)
}

func (d TensorMeasurementDocument) Content() (artifact.Content, error) {
	return tensorMeasurementCodec.Content(d)
}

func (d TensorMeasurementDocument) Lineage() artifact.Lineage {
	return artifact.Lineage{Child: d.ID, Parent: d.Inventory, Relation: artifact.RelationDerivedFrom}
}

func (d TensorMeasurementDocument) Batch(key string) (artifact.Batch, error) {
	content, err := d.Content()
	if err != nil {
		return artifact.Batch{}, err
	}
	return artifact.NewDocumentBatch(key, []artifact.Content{content}, []artifact.Lineage{d.Lineage()}, nil)
}

func measurementFromSamples(name string, elements uint64, samples []float64) (TensorMeasurement, error) {
	characterization, ok := tensorstats.Characterize(elements, samples)
	if !ok {
		return TensorMeasurement{}, errors.New("model artifact: tensor sample has no finite values")
	}
	return TensorMeasurement{Name: name, Characterization: characterization}, nil
}

func canonicalizeTensorMeasurements(document *TensorMeasurementDocument) error {
	if document == nil || document.Version != TensorMeasurementVersion ||
		document.Inventory.Kind() != artifact.KindTensorInventory ||
		document.Policy.MaxSamplesPerTensor == 0 ||
		document.Policy.MaxReadBytes == 0 || document.ReadBytes == 0 ||
		document.ReadBytes > document.Policy.MaxReadBytes || len(document.Measurements) == 0 ||
		len(document.Measurements) > maxInventoryTensors {
		return errors.New("model artifact: invalid tensor measurement envelope")
	}
	sort.Slice(document.Measurements, func(i, j int) bool {
		return document.Measurements[i].Name < document.Measurements[j].Name
	})
	for index, measurement := range document.Measurements {
		c := measurement.Characterization
		if measurement.Name == "" || strings.TrimSpace(measurement.Name) != measurement.Name ||
			c.Elements == 0 || c.Samples == 0 || c.Samples > c.Elements ||
			c.Samples > document.Policy.MaxSamplesPerTensor ||
			c.FiniteSamples == 0 || c.FiniteSamples > c.Samples ||
			!finiteMeasurement(c.LowerQuartile) || !finiteMeasurement(c.Median) ||
			!finiteMeasurement(c.UpperQuartile) || c.InterquartileRange < 0 ||
			c.LowerQuartile > c.Median || c.Median > c.UpperQuartile ||
			index > 0 && document.Measurements[index-1].Name == measurement.Name {
			return errors.New("model artifact: invalid tensor measurement")
		}
	}
	return nil
}

func finiteMeasurement(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func tensorMeasurementContent(document TensorMeasurementDocument) ([]byte, error) {
	document.ID = artifact.ID{}
	content, err := json.Marshal(document)
	if err != nil {
		return nil, fmt.Errorf("model artifact: encode tensor measurement: %w", err)
	}
	return content, nil
}
