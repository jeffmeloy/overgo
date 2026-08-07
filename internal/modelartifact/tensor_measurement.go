package modelartifact

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

const (
	TensorMeasurementVersion   uint16 = 1
	TensorMeasurementMediaType        = "application/vnd.overgo.tensor-measurement+json"
	TensorMeasurementSchema           = "overgo/tensor-measurement/v1"

	defaultMeasurementSamples   = 4096
	defaultMeasurementReadBytes = 256 << 20
)

var tensorMeasurementContract = artifact.DocumentContract{
	Kind: artifact.KindTensorInventory, MediaType: TensorMeasurementMediaType, Schema: TensorMeasurementSchema,
}

type MeasurementPolicy struct {
	MaxSamplesPerTensor uint64 `json:"max_samples_per_tensor"`
	MaxReadBytes        uint64 `json:"max_read_bytes"`
}

func DefaultMeasurementPolicy() MeasurementPolicy {
	return MeasurementPolicy{
		MaxSamplesPerTensor: defaultMeasurementSamples,
		MaxReadBytes:        defaultMeasurementReadBytes,
	}
}

type TensorMeasurement struct {
	Name             string  `json:"name"`
	Elements         uint64  `json:"elements"`
	Samples          uint64  `json:"samples"`
	FiniteSamples    uint64  `json:"finite_samples"`
	NonFiniteSamples uint64  `json:"non_finite_samples"`
	P05              float64 `json:"p05"`
	Median           float64 `json:"median"`
	P95              float64 `json:"p95"`
	MAD              float64 `json:"mad"`
}

func (m TensorMeasurement) SampleFraction() float64 {
	if m.Elements == 0 {
		return 0
	}
	return float64(m.Samples) / float64(m.Elements)
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

func NewTensorMeasurementDocument(
	inventory artifact.ID,
	policy MeasurementPolicy,
	readBytes uint64,
	measurements []TensorMeasurement,
) (TensorMeasurementDocument, error) {
	document := TensorMeasurementDocument{
		Version: TensorMeasurementVersion, Inventory: inventory, Policy: policy,
		ReadBytes: readBytes, Measurements: slices.Clone(measurements),
	}
	if err := canonicalizeTensorMeasurements(&document); err != nil {
		return TensorMeasurementDocument{}, err
	}
	content, err := tensorMeasurementContent(document)
	if err != nil {
		return TensorMeasurementDocument{}, err
	}
	document.ID, err = tensorMeasurementContract.Identify(content)
	return document, err
}

func ParseTensorMeasurementDocument(content []byte) (TensorMeasurementDocument, error) {
	var body TensorMeasurementDocument
	if err := strictjson.DecodeBytes(content, &body); err != nil {
		return TensorMeasurementDocument{}, fmt.Errorf("model artifact: decode tensor measurement: %w", err)
	}
	document, err := NewTensorMeasurementDocument(body.Inventory, body.Policy, body.ReadBytes, body.Measurements)
	if err != nil {
		return TensorMeasurementDocument{}, err
	}
	canonical, err := document.ContentBytes()
	if err != nil {
		return TensorMeasurementDocument{}, err
	}
	if !bytes.Equal(canonical, content) {
		return TensorMeasurementDocument{}, errors.New("model artifact: non-canonical tensor measurement")
	}
	return document, nil
}

func (d TensorMeasurementDocument) ValidateIdentity() error {
	if d.ID.Kind() != artifact.KindTensorInventory {
		return errors.New("model artifact: invalid tensor measurement identity")
	}
	canonical := d
	canonical.ID = artifact.ID{}
	canonical.Measurements = slices.Clone(d.Measurements)
	if err := canonicalizeTensorMeasurements(&canonical); err != nil {
		return err
	}
	canonical.ID = d.ID
	if d.Version != canonical.Version || d.Inventory != canonical.Inventory || d.Policy != canonical.Policy ||
		d.ReadBytes != canonical.ReadBytes || !slices.Equal(d.Measurements, canonical.Measurements) {
		return errors.New("model artifact: tensor measurement is not canonical")
	}
	canonical.ID = artifact.ID{}
	content, err := tensorMeasurementContent(canonical)
	if err != nil {
		return err
	}
	if err := tensorMeasurementContract.ValidateIdentity(d.ID, content); err != nil {
		return errors.New("model artifact: tensor measurement identity mismatch")
	}
	return nil
}

func (d TensorMeasurementDocument) ContentBytes() ([]byte, error) {
	if err := d.ValidateIdentity(); err != nil {
		return nil, err
	}
	d.ID = artifact.ID{}
	return tensorMeasurementContent(d)
}

func (d TensorMeasurementDocument) Content() (artifact.Content, error) {
	id := d.ID
	content, err := d.ContentBytes()
	if err != nil {
		return artifact.Content{}, err
	}
	return tensorMeasurementContract.Content(id, content)
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
	measurement := TensorMeasurement{Name: name, Elements: elements, Samples: uint64(len(samples))}
	finite := make([]float64, 0, len(samples))
	for _, value := range samples {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			measurement.NonFiniteSamples++
		} else {
			finite = append(finite, value)
		}
	}
	measurement.FiniteSamples = uint64(len(finite))
	if len(finite) == 0 {
		return TensorMeasurement{}, errors.New("model artifact: tensor sample has no finite values")
	}
	sort.Float64s(finite)
	measurement.P05 = sampleQuantile(finite, 5, 100)
	measurement.Median = sampleQuantile(finite, 1, 2)
	measurement.P95 = sampleQuantile(finite, 95, 100)
	deviations := make([]float64, len(finite))
	for index, value := range finite {
		deviations[index] = math.Abs(value - measurement.Median)
	}
	sort.Float64s(deviations)
	measurement.MAD = sampleQuantile(deviations, 1, 2)
	return measurement, nil
}

func sampleQuantile(sorted []float64, numerator, denominator uint64) float64 {
	index := uint64(len(sorted)-1) * numerator / denominator
	return sorted[index]
}

func canonicalizeTensorMeasurements(document *TensorMeasurementDocument) error {
	if document == nil || document.Version != TensorMeasurementVersion ||
		document.Inventory.Kind() != artifact.KindTensorInventory ||
		document.Policy.MaxSamplesPerTensor < 256 || document.Policy.MaxSamplesPerTensor > 1<<20 ||
		document.Policy.MaxReadBytes == 0 || document.ReadBytes == 0 ||
		document.ReadBytes > document.Policy.MaxReadBytes || len(document.Measurements) == 0 ||
		len(document.Measurements) > maxInventoryTensors {
		return errors.New("model artifact: invalid tensor measurement envelope")
	}
	sort.Slice(document.Measurements, func(i, j int) bool {
		return document.Measurements[i].Name < document.Measurements[j].Name
	})
	for index, measurement := range document.Measurements {
		if measurement.Name == "" || strings.TrimSpace(measurement.Name) != measurement.Name ||
			measurement.Elements == 0 || measurement.Samples == 0 || measurement.Samples > measurement.Elements ||
			measurement.Samples > document.Policy.MaxSamplesPerTensor ||
			measurement.FiniteSamples+measurement.NonFiniteSamples != measurement.Samples ||
			measurement.FiniteSamples == 0 || !finiteMeasurement(measurement.P05) ||
			!finiteMeasurement(measurement.Median) || !finiteMeasurement(measurement.P95) ||
			!finiteMeasurement(measurement.MAD) || measurement.MAD < 0 ||
			measurement.P05 > measurement.Median || measurement.Median > measurement.P95 ||
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
