// Package adaptiveparity owns the neutral import contract for reference
// capabilities and evidence. Imported snapshots become content-addressed Overgo
// documents; no runtime code imports or calls the source repository.
package adaptiveparity

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

const (
	Version          uint16 = 1
	Schema                  = "overgo/adaptive-parity-snapshot/v1"
	MediaType               = "application/vnd.overgo.adaptive-parity-snapshot+json"
	maxSnapshotBytes        = 4 << 20
)

type Modality string

const (
	ModalityText       Modality = "text"
	ModalityImage      Modality = "image"
	ModalityAudio      Modality = "audio"
	ModalityVideo      Modality = "video"
	ModalityTimeSeries Modality = "time-series"
	ModalityTable      Modality = "table"
)

type Source struct {
	Name       string `json:"name"`
	Repository string `json:"repository"`
	Commit     string `json:"commit"`
}

type Reference struct {
	Name     string      `json:"name"`
	Identity artifact.ID `json:"identity"`
}

type Signature struct {
	Inputs  []Modality `json:"inputs"`
	Outputs []Modality `json:"outputs"`
}

type Measurement struct {
	Runtime   string      `json:"runtime"`
	Protocol  string      `json:"protocol"`
	WallNanos uint64      `json:"wall_nanos,omitempty"`
	PeakBytes uint64      `json:"peak_bytes,omitempty"`
	Evidence  artifact.ID `json:"evidence"`
}

type Capability struct {
	ID           string        `json:"id"`
	Source       string        `json:"source"`
	EntryPoint   string        `json:"entry_point"`
	Signature    Signature     `json:"signature"`
	Artifacts    []Reference   `json:"artifacts"`
	Corpora      []Reference   `json:"corpora"`
	Goldens      []Reference   `json:"goldens"`
	Measurements []Measurement `json:"measurements,omitempty"`
}

type Snapshot struct {
	Version      uint16       `json:"version"`
	Sources      []Source     `json:"sources"`
	Capabilities []Capability `json:"capabilities"`
	ID           artifact.ID  `json:"-"`
}

var codec = artifact.DocumentCodec[Snapshot]{
	Name: "adaptive parity snapshot",
	Contract: artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: MediaType, Schema: Schema,
	},
	Decode: func(data []byte, snapshot *Snapshot) error { return strictjson.DecodeBytes(data, snapshot) },
	Encode: func(snapshot Snapshot) ([]byte, error) {
		snapshot.ID = artifact.ID{}
		return json.Marshal(snapshot)
	},
	Canonicalize: canonicalize,
	Clone:        clone,
	Identity:     func(snapshot Snapshot) artifact.ID { return snapshot.ID },
	SetIdentity:  func(snapshot *Snapshot, id artifact.ID) { snapshot.ID = id },
}

func New(sources []Source, capabilities []Capability) (Snapshot, error) {
	return codec.New(Snapshot{Version: Version, Sources: slices.Clone(sources), Capabilities: slices.Clone(capabilities)})
}

func Parse(data []byte) (Snapshot, error)             { return codec.Parse(data) }
func Normalize(data []byte) (Snapshot, []byte, error) { return codec.Normalize(data) }
func (s Snapshot) Content() (artifact.Content, error) { return codec.Content(s) }
func (s Snapshot) ValidateIdentity() error            { return codec.ValidateIdentity(s) }

// Import admits one bounded snapshot and returns its canonical stored bytes.
func Import(reader io.Reader) (Snapshot, []byte, error) {
	if reader == nil {
		return Snapshot{}, nil, errors.New("adaptive parity: nil snapshot reader")
	}
	data, err := io.ReadAll(io.LimitReader(reader, maxSnapshotBytes+1))
	if err != nil {
		return Snapshot{}, nil, fmt.Errorf("adaptive parity: read snapshot: %w", err)
	}
	if len(data) > maxSnapshotBytes {
		return Snapshot{}, nil, errors.New("adaptive parity: snapshot exceeds limit")
	}
	return Normalize(data)
}

func canonicalize(snapshot *Snapshot) error {
	if snapshot == nil || snapshot.Version != Version || len(snapshot.Sources) == 0 || len(snapshot.Capabilities) == 0 {
		return errors.New("adaptive parity: invalid snapshot")
	}
	slices.SortFunc(snapshot.Sources, func(a, b Source) int { return strings.Compare(a.Name, b.Name) })
	sourceNames := make(map[string]struct{}, len(snapshot.Sources))
	for _, source := range snapshot.Sources {
		if !validName(source.Name) || !validText(source.Repository) || !validCommit(source.Commit) {
			return errors.New("adaptive parity: invalid source")
		}
		if _, exists := sourceNames[source.Name]; exists {
			return fmt.Errorf("adaptive parity: duplicate source %q", source.Name)
		}
		sourceNames[source.Name] = struct{}{}
	}
	slices.SortFunc(snapshot.Capabilities, func(a, b Capability) int { return strings.Compare(a.ID, b.ID) })
	capabilityIDs := make(map[string]struct{}, len(snapshot.Capabilities))
	for index := range snapshot.Capabilities {
		capability := &snapshot.Capabilities[index]
		if !validName(capability.ID) || !validText(capability.EntryPoint) || strings.Contains(capability.EntryPoint, "\\") {
			return errors.New("adaptive parity: invalid capability")
		}
		if _, exists := sourceNames[capability.Source]; !exists {
			return fmt.Errorf("adaptive parity: capability %q names unknown source %q", capability.ID, capability.Source)
		}
		if _, exists := capabilityIDs[capability.ID]; exists {
			return fmt.Errorf("adaptive parity: duplicate capability %q", capability.ID)
		}
		capabilityIDs[capability.ID] = struct{}{}
		if err := validateSignature(capability.Signature); err != nil {
			return fmt.Errorf("adaptive parity: capability %q: %w", capability.ID, err)
		}
		for _, references := range []*[]Reference{&capability.Artifacts, &capability.Corpora, &capability.Goldens} {
			if err := canonicalReferences(references); err != nil {
				return fmt.Errorf("adaptive parity: capability %q: %w", capability.ID, err)
			}
		}
		if len(capability.Artifacts) == 0 || len(capability.Corpora) == 0 || len(capability.Goldens) == 0 {
			return fmt.Errorf("adaptive parity: capability %q lacks artifact, corpus, or golden evidence", capability.ID)
		}
		slices.SortFunc(capability.Measurements, func(a, b Measurement) int {
			if order := strings.Compare(a.Runtime, b.Runtime); order != 0 {
				return order
			}
			return strings.Compare(a.Protocol, b.Protocol)
		})
		for _, measurement := range capability.Measurements {
			if !validName(measurement.Runtime) || !validText(measurement.Protocol) ||
				measurement.WallNanos == 0 && measurement.PeakBytes == 0 || measurement.Evidence.Kind() != artifact.KindEvidence {
				return fmt.Errorf("adaptive parity: capability %q has invalid measurement", capability.ID)
			}
		}
	}
	return nil
}

func canonicalReferences(references *[]Reference) error {
	slices.SortFunc(*references, func(a, b Reference) int { return strings.Compare(a.Name, b.Name) })
	seen := make(map[string]struct{}, len(*references))
	for _, reference := range *references {
		if !validName(reference.Name) || !reference.Identity.Valid() {
			return errors.New("invalid reference")
		}
		if _, exists := seen[reference.Name]; exists {
			return fmt.Errorf("duplicate reference %q", reference.Name)
		}
		seen[reference.Name] = struct{}{}
	}
	return nil
}

func validateSignature(signature Signature) error {
	if len(signature.Inputs) == 0 || len(signature.Outputs) == 0 {
		return errors.New("empty modality signature")
	}
	for _, modalities := range [][]Modality{signature.Inputs, signature.Outputs} {
		for _, modality := range modalities {
			switch modality {
			case ModalityText, ModalityImage, ModalityAudio, ModalityVideo, ModalityTimeSeries, ModalityTable:
			default:
				return fmt.Errorf("invalid modality %q", modality)
			}
		}
	}
	return nil
}

func clone(snapshot Snapshot) Snapshot {
	snapshot.Sources = slices.Clone(snapshot.Sources)
	snapshot.Capabilities = slices.Clone(snapshot.Capabilities)
	for index := range snapshot.Capabilities {
		capability := &snapshot.Capabilities[index]
		capability.Signature.Inputs = slices.Clone(capability.Signature.Inputs)
		capability.Signature.Outputs = slices.Clone(capability.Signature.Outputs)
		capability.Artifacts = slices.Clone(capability.Artifacts)
		capability.Corpora = slices.Clone(capability.Corpora)
		capability.Goldens = slices.Clone(capability.Goldens)
		capability.Measurements = slices.Clone(capability.Measurements)
	}
	return snapshot
}

func validCommit(value string) bool {
	if len(value) != 40 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func validName(value string) bool {
	return value != "" && len(value) <= 128 && strings.TrimSpace(value) == value && !strings.ContainsAny(value, "\x00\r/\\")
}

func validText(value string) bool {
	return value != "" && len(value) <= 4096 && strings.TrimSpace(value) == value && !strings.ContainsAny(value, "\x00\r")
}
