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
	"overgo/internal/recipecontract"
	"overgo/internal/strictjson"
	"overgo/internal/textcheck"
)

const (
	Version          uint16 = 2
	Schema                  = "overgo/adaptive-parity-snapshot/v2"
	MediaType               = "application/vnd.overgo.adaptive-parity-snapshot+json"
	maxSnapshotBytes        = 4 << 20
)

type Modality = recipecontract.Modality

const (
	ModalityText       = recipecontract.ModalityText
	ModalityImage      = recipecontract.ModalityImage
	ModalityAudio      = recipecontract.ModalityAudio
	ModalityVideo      = recipecontract.ModalityVideo
	ModalityTimeSeries = recipecontract.ModalityTimeSeries
	ModalityTable      = recipecontract.ModalityTable
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

type Signature = recipecontract.ModalitySignature

type Measurement struct {
	Role      MeasurementRole `json:"role"`
	Runtime   string          `json:"runtime"`
	Protocol  string          `json:"protocol"`
	WallNanos uint64          `json:"wall_nanos,omitempty"`
	PeakBytes uint64          `json:"peak_bytes,omitempty"`
	Evidence  artifact.ID     `json:"evidence"`
}

type MeasurementRole string

const (
	MeasurementReference MeasurementRole = "reference"
	MeasurementCandidate MeasurementRole = "candidate"
)

type PromotionState string

const (
	PromotionRefused  PromotionState = "refused"
	PromotionParity   PromotionState = "parity"
	PromotionWallLead PromotionState = "wall-lead"
	PromotionLead     PromotionState = "lead"
	PromotionTradeoff PromotionState = "tradeoff"
)

type Capability struct {
	ID           string         `json:"id"`
	Source       string         `json:"source"`
	EntryPoint   string         `json:"entry_point"`
	Signature    Signature      `json:"signature"`
	State        PromotionState `json:"state"`
	Refusal      string         `json:"refusal,omitempty"`
	Artifacts    []Reference    `json:"artifacts"`
	Corpora      []Reference    `json:"corpora"`
	Goldens      []Reference    `json:"goldens"`
	Measurements []Measurement  `json:"measurements,omitempty"`
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
		if !textcheck.Bounded(source.Name, 128, "\x00\r/\\") || !textcheck.Bounded(source.Repository, 4096, "\x00\r") || !validCommit(source.Commit) {
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
		if !textcheck.Bounded(capability.ID, 128, "\x00\r/\\") || !textcheck.Bounded(capability.EntryPoint, 4096, "\x00\r") || strings.Contains(capability.EntryPoint, "\\") {
			return errors.New("adaptive parity: invalid capability")
		}
		if _, exists := sourceNames[capability.Source]; !exists {
			return fmt.Errorf("adaptive parity: capability %q names unknown source %q", capability.ID, capability.Source)
		}
		if _, exists := capabilityIDs[capability.ID]; exists {
			return fmt.Errorf("adaptive parity: duplicate capability %q", capability.ID)
		}
		capabilityIDs[capability.ID] = struct{}{}
		if err := capability.Signature.Validate(); err != nil {
			return fmt.Errorf("adaptive parity: capability %q: %w", capability.ID, err)
		}
		for _, references := range []*[]Reference{&capability.Artifacts, &capability.Corpora, &capability.Goldens} {
			if err := canonicalReferences(references); err != nil {
				return fmt.Errorf("adaptive parity: capability %q: %w", capability.ID, err)
			}
		}
		if err := validatePromotionEvidence(*capability); err != nil {
			return fmt.Errorf("adaptive parity: capability %q: %w", capability.ID, err)
		}
		slices.SortFunc(capability.Measurements, func(a, b Measurement) int {
			if order := strings.Compare(a.Runtime, b.Runtime); order != 0 {
				return order
			}
			return strings.Compare(a.Protocol, b.Protocol)
		})
		for _, measurement := range capability.Measurements {
			if !validMeasurementRole(measurement.Role) || !textcheck.Bounded(measurement.Runtime, 128, "\x00\r/\\") ||
				!textcheck.Bounded(measurement.Protocol, 4096, "\x00\r") ||
				measurement.WallNanos == 0 && measurement.PeakBytes == 0 || measurement.Evidence.Kind() != artifact.KindEvidence {
				return fmt.Errorf("adaptive parity: capability %q has invalid measurement", capability.ID)
			}
		}
		if err := validateMeasurementVerdict(*capability); err != nil {
			return fmt.Errorf("adaptive parity: capability %q: %w", capability.ID, err)
		}
	}
	return nil
}

func validatePromotionEvidence(capability Capability) error {
	switch capability.State {
	case PromotionRefused:
		if !textcheck.Bounded(capability.Refusal, 4096, "\x00\r") {
			return errors.New("refusal lacks reason")
		}
		if len(capability.Artifacts)+len(capability.Corpora)+len(capability.Goldens)+len(capability.Measurements) != 0 {
			return errors.New("refusal carries support evidence")
		}
	case PromotionParity, PromotionWallLead, PromotionLead, PromotionTradeoff:
		if capability.Refusal != "" {
			return errors.New("supported capability carries refusal")
		}
		if len(capability.Artifacts) == 0 || len(capability.Corpora) == 0 || len(capability.Goldens) == 0 {
			return errors.New("lacks artifact, corpus, or golden evidence")
		}
	default:
		return errors.New("invalid promotion state")
	}
	return nil
}

func validMeasurementRole(role MeasurementRole) bool {
	return role == MeasurementReference || role == MeasurementCandidate
}

func validateMeasurementVerdict(capability Capability) error {
	if capability.State == PromotionRefused {
		return nil
	}
	type pair struct{ reference, candidate *Measurement }
	pairs := make(map[string]pair)
	for index := range capability.Measurements {
		measurement := &capability.Measurements[index]
		current := pairs[measurement.Protocol]
		if measurement.Role == MeasurementReference {
			if current.reference != nil {
				return fmt.Errorf("duplicate reference measurement for %q", measurement.Protocol)
			}
			current.reference = measurement
		} else {
			if current.candidate != nil {
				return fmt.Errorf("duplicate candidate measurement for %q", measurement.Protocol)
			}
			current.candidate = measurement
		}
		pairs[measurement.Protocol] = current
	}
	if capability.State == PromotionParity {
		return nil
	}
	if len(pairs) == 0 {
		return errors.New("measured verdict lacks measurements")
	}
	for protocol, measurements := range pairs {
		if measurements.reference == nil || measurements.candidate == nil {
			return fmt.Errorf("protocol %q lacks matched measurements", protocol)
		}
		reference, candidate := measurements.reference, measurements.candidate
		wallLead := reference.WallNanos > 0 && candidate.WallNanos > 0 && candidate.WallNanos <= reference.WallNanos
		peakLead := reference.PeakBytes > 0 && candidate.PeakBytes > 0 && candidate.PeakBytes <= reference.PeakBytes
		switch capability.State {
		case PromotionWallLead:
			if !wallLead || reference.PeakBytes != 0 || candidate.PeakBytes != 0 {
				return fmt.Errorf("protocol %q does not prove a wall-only lead", protocol)
			}
		case PromotionLead:
			if !wallLead || !peakLead {
				return fmt.Errorf("protocol %q does not prove wall and peak leadership", protocol)
			}
		case PromotionTradeoff:
			if reference.WallNanos == 0 || candidate.WallNanos == 0 || reference.PeakBytes == 0 || candidate.PeakBytes == 0 || wallLead == peakLead {
				return fmt.Errorf("protocol %q does not prove a wall/peak tradeoff", protocol)
			}
		}
	}
	return nil
}

func canonicalReferences(references *[]Reference) error {
	slices.SortFunc(*references, func(a, b Reference) int { return strings.Compare(a.Name, b.Name) })
	seen := make(map[string]struct{}, len(*references))
	for _, reference := range *references {
		if !textcheck.Bounded(reference.Name, 128, "\x00\r/\\") || !reference.Identity.Valid() {
			return errors.New("invalid reference")
		}
		if _, exists := seen[reference.Name]; exists {
			return fmt.Errorf("duplicate reference %q", reference.Name)
		}
		seen[reference.Name] = struct{}{}
	}
	return nil
}

func clone(snapshot Snapshot) Snapshot {
	snapshot.Sources = slices.Clone(snapshot.Sources)
	snapshot.Capabilities = slices.Clone(snapshot.Capabilities)
	for index := range snapshot.Capabilities {
		capability := &snapshot.Capabilities[index]
		capability.Signature = capability.Signature.Clone()
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
