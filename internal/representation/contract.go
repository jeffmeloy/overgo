// Package representation owns cross-model tensor interface authority.
package representation

import (
	"errors"
	"fmt"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

const (
	// ContractVersion is the current representation contract document version.
	ContractVersion uint16 = artifact.InitialDocumentVersion
	// ContractMediaType identifies serialized representation contracts.
	ContractMediaType = "application/vnd.overgo.representation-contract+json"
	// ContractSchema identifies the canonical representation contract schema.
	ContractSchema = "overgo/representation-contract/v1"
)

// TapPoint identifies a model boundary at which a representation is observed.
type TapPoint string

const (
	// TapEmbeddingOutput selects the output of a model's embedding stage.
	TapEmbeddingOutput TapPoint = "embedding-output"
	// TapEncoderOutput selects the output of a model's encoder.
	TapEncoderOutput TapPoint = "encoder-output"
	// TapLayerInput selects the input boundary of a numbered layer.
	TapLayerInput TapPoint = "layer-input"
	// TapLayerOutput selects the output boundary of a numbered layer.
	TapLayerOutput TapPoint = "layer-output"
	// TapAttentionInput selects the attention input of a numbered layer.
	TapAttentionInput TapPoint = "attention-input"
)

// Modality identifies the semantic domain represented by a tensor interface.
type Modality string

const (
	// ModalityText identifies textual representations.
	ModalityText Modality = "text"
	// ModalityImage identifies image representations.
	ModalityImage Modality = "image"
	// ModalityAudio identifies audio representations.
	ModalityAudio Modality = "audio"
	// ModalityVideo identifies video representations.
	ModalityVideo Modality = "video"
	// ModalityAction identifies action representations.
	ModalityAction Modality = "action"
	// ModalitySeries identifies time-series representations.
	ModalitySeries Modality = "series"
	// ModalityTabular identifies tabular representations.
	ModalityTabular Modality = "tabular"
	// ModalityLatent identifies modality-neutral latent representations.
	ModalityLatent Modality = "latent"
)

// AxisKind identifies the semantic meaning of a tensor dimension.
type AxisKind string

const (
	// AxisChannel identifies a feature-channel dimension.
	AxisChannel AxisKind = "channel"
	// AxisSequence identifies an ordered token or sample dimension.
	AxisSequence AxisKind = "sequence"
	// AxisBatch identifies an independent batch dimension.
	AxisBatch AxisKind = "batch"
	// AxisWidth identifies a spatial width dimension.
	AxisWidth AxisKind = "width"
	// AxisHeight identifies a spatial height dimension.
	AxisHeight AxisKind = "height"
	// AxisTime identifies a temporal dimension.
	AxisTime AxisKind = "time"
)

// AxisBounds is either one exact Extent or an inclusive dynamic range.
type AxisBounds struct {
	Extent  uint64 `json:"extent,omitempty"`
	Minimum uint64 `json:"minimum,omitempty"`
	Maximum uint64 `json:"maximum,omitempty"`
}

// Axis binds semantic meaning to exact or bounded dimension geometry.
type Axis struct {
	Kind   AxisKind   `json:"kind"`
	Bounds AxisBounds `json:"bounds"`
}

// TensorContract describes the data type and ordered axes of a representation.
type TensorContract struct {
	DataType dtype.Type `json:"data_type"`
	Axes     []Axis     `json:"axes"`
}

// MaskPolicy defines how invalid sequence positions are represented.
type MaskPolicy string

const (
	// MaskNone declares that every sequence position is valid.
	MaskNone MaskPolicy = "none"
	// MaskPrefix declares a contiguous valid prefix followed by padding.
	MaskPrefix MaskPolicy = "valid-prefix"
	// MaskExplicit declares validity through an explicit mask.
	MaskExplicit MaskPolicy = "explicit"
)

// PaddingPolicy defines where padding may occur in a sequence.
type PaddingPolicy string

const (
	// PaddingNone declares that the sequence contains no padding.
	PaddingNone PaddingPolicy = "none"
	// PaddingSuffix declares that padding follows the valid sequence prefix.
	PaddingSuffix PaddingPolicy = "suffix"
)

// PositionPolicy defines how positions are assigned to representation elements.
type PositionPolicy string

const (
	// PositionNone declares that the representation carries no position semantics.
	PositionNone PositionPolicy = "none"
	// PositionSequential declares positions along one sequence axis.
	PositionSequential PositionPolicy = "sequential"
	// PositionMultiAxis declares positions across multiple tensor axes.
	PositionMultiAxis PositionPolicy = "multi-axis"
)

// SpecialTokenRole identifies a model-defined structural token.
type SpecialTokenRole string

const (
	// SpecialTokenBOS identifies a beginning-of-sequence token.
	SpecialTokenBOS SpecialTokenRole = "bos"
	// SpecialTokenEOS identifies an end-of-sequence token.
	SpecialTokenEOS SpecialTokenRole = "eos"
	// SpecialTokenCLS identifies a classification token.
	SpecialTokenCLS SpecialTokenRole = "cls"
	// SpecialTokenSeparator identifies a sequence separator token.
	SpecialTokenSeparator SpecialTokenRole = "separator"
)

// TokenDisposition defines whether a structural token survives a bridge.
type TokenDisposition string

const (
	// TokenKeep preserves a structural token.
	TokenKeep TokenDisposition = "keep"
	// TokenDrop removes a structural token.
	TokenDrop TokenDisposition = "drop"
)

// SpecialToken binds a structural role to its bridge disposition.
type SpecialToken struct {
	Role        SpecialTokenRole `json:"role"`
	Disposition TokenDisposition `json:"disposition"`
}

// SequenceContract defines masking, padding, positioning, and structural tokens.
type SequenceContract struct {
	Axis          AxisKind       `json:"axis"`
	Mask          MaskPolicy     `json:"mask"`
	Padding       PaddingPolicy  `json:"padding"`
	Position      PositionPolicy `json:"position"`
	PositionAxes  []AxisKind     `json:"position_axes,omitempty"`
	SpecialTokens []SpecialToken `json:"special_tokens,omitempty"`
}

// NormalizationKind identifies the normalization applied at an interface.
type NormalizationKind string

const (
	// NormalizationNone declares an unnormalized representation.
	NormalizationNone NormalizationKind = "none"
	// NormalizationRMS declares root-mean-square normalization.
	NormalizationRMS NormalizationKind = "rms"
	// NormalizationLayer declares layer normalization.
	NormalizationLayer NormalizationKind = "layer"
	// NormalizationL2 declares Euclidean unit normalization.
	NormalizationL2 NormalizationKind = "l2"
)

// MagnitudePolicy defines how a bridge treats representation magnitude.
type MagnitudePolicy string

const (
	// MagnitudeNative preserves the target interface's native magnitude convention.
	MagnitudeNative MagnitudePolicy = "native"
	// MagnitudeUnit requires a unit-magnitude representation.
	MagnitudeUnit MagnitudePolicy = "unit"
	// MagnitudePreserve retains the source representation's magnitude.
	MagnitudePreserve MagnitudePolicy = "preserve-source"
)

// NormalizationContract defines normalization kind, guard, and magnitude semantics.
type NormalizationContract struct {
	Kind      NormalizationKind `json:"kind"`
	Epsilon   float32           `json:"epsilon,omitempty"`
	Magnitude MagnitudePolicy   `json:"magnitude"`
}

// AuthorityRole identifies an artifact responsible for representation semantics.
type AuthorityRole string

const (
	// AuthorityTokenizer assigns token identity and vocabulary semantics.
	AuthorityTokenizer AuthorityRole = "tokenizer"
	// AuthorityProjector assigns modality projection semantics.
	AuthorityProjector AuthorityRole = "projector"
	// AuthorityProcessor assigns preprocessing semantics.
	AuthorityProcessor AuthorityRole = "processor"
	// AuthorityProfile assigns model profile semantics.
	AuthorityProfile AuthorityRole = "profile"
	// AuthorityAdapter assigns bridge adaptation semantics.
	AuthorityAdapter AuthorityRole = "adapter"
)

// Authority binds one semantic role to an immutable artifact.
type Authority struct {
	Role     AuthorityRole `json:"role"`
	Artifact artifact.ID   `json:"artifact"`
}

// Producer identifies the exact model boundary that emits a representation.
type Producer struct {
	Model      artifact.ID `json:"model"`
	Definition artifact.ID `json:"definition"`
	Tap        TapPoint    `json:"tap"`
	Layer      *uint32     `json:"layer,omitempty"`
}

// Contract is the complete interface one producer exposes to a bridge.
type Contract struct {
	Version       uint16                `json:"version"`
	Producer      Producer              `json:"producer"`
	Modality      Modality              `json:"modality"`
	Tensor        TensorContract        `json:"tensor"`
	Sequence      SequenceContract      `json:"sequence"`
	Normalization NormalizationContract `json:"normalization"`
	Authorities   []Authority           `json:"authorities,omitempty"`
	ID            artifact.ID           `json:"-"`
}

var contractCodec = artifact.JSONDocumentCodec(
	"representation contract", artifact.KindProfile, ContractMediaType, ContractSchema,
	canonicalizeContract,
	func(value Contract) artifact.ID { return value.ID },
	func(value *Contract, id artifact.ID) { value.ID = id },
	cloneContract,
)

// ParseContract admits canonical representation authority bytes.
func ParseContract(content []byte) (Contract, error) { return contractCodec.Parse(content) }

// ValidateIdentity verifies canonical content and its stored identity.
func (c Contract) ValidateIdentity() error { return contractCodec.ValidateIdentity(c) }

// Content returns the canonical serialized representation contract.
func (c Contract) Content() (artifact.Content, error) { return contractCodec.Content(c) }

// Lineage returns the model, definition, and semantic authorities bound by the contract.
func (c Contract) Lineage() []artifact.Lineage {
	parents := []artifact.ID{c.Producer.Model, c.Producer.Definition}
	for _, authority := range c.Authorities {
		parents = append(parents, authority.Artifact)
	}
	return artifact.DependencyLineage(c.ID, parents...)
}

// Batch returns an atomic publication containing the contract and its lineage.
func (c Contract) Batch(key string) (artifact.Batch, error) {
	return contractCodec.Batch(key, c, c.Lineage(), nil)
}

func cloneContract(value Contract) Contract {
	value.Tensor.Axes = slices.Clone(value.Tensor.Axes)
	value.Sequence.PositionAxes = slices.Clone(value.Sequence.PositionAxes)
	value.Sequence.SpecialTokens = slices.Clone(value.Sequence.SpecialTokens)
	value.Authorities = slices.Clone(value.Authorities)
	if value.Producer.Layer != nil {
		layer := *value.Producer.Layer
		value.Producer.Layer = &layer
	}
	return value
}

func canonicalizeContract(value *Contract) error {
	if value == nil || value.Version != ContractVersion {
		return errors.New("representation: invalid contract version")
	}
	if err := validateProducer(value.Producer); err != nil {
		return err
	}
	if !value.Modality.valid() {
		return errors.New("representation: invalid modality")
	}
	if err := validateTensor(value.Tensor); err != nil {
		return err
	}
	if err := validateSequence(value.Sequence, value.Tensor.Axes); err != nil {
		return err
	}
	if err := validateNormalization(value.Normalization); err != nil {
		return err
	}
	sort.Slice(value.Authorities, func(i, j int) bool {
		if value.Authorities[i].Role == value.Authorities[j].Role {
			return value.Authorities[i].Artifact.String() < value.Authorities[j].Artifact.String()
		}
		return value.Authorities[i].Role < value.Authorities[j].Role
	})
	for index, authority := range value.Authorities {
		if err := validateAuthority(authority); err != nil {
			return err
		}
		if index > tensor.FirstOffset && value.Authorities[index-tensor.SingletonExtent].Role == authority.Role {
			return errors.New("representation: duplicate authority role")
		}
	}
	return nil
}

func validateProducer(value Producer) error {
	if value.Model.Kind() != artifact.KindModel || value.Definition.Kind() != artifact.KindModelDefinition {
		return errors.New("representation: producer requires exact model and definition authorities")
	}
	switch value.Tap {
	case TapEmbeddingOutput, TapEncoderOutput:
		if value.Layer != nil {
			return errors.New("representation: model-boundary tap cannot select a layer")
		}
	case TapLayerInput, TapLayerOutput, TapAttentionInput:
		if value.Layer == nil {
			return errors.New("representation: layer tap requires a layer")
		}
	default:
		return errors.New("representation: invalid producer tap")
	}
	return nil
}

func validateTensor(value TensorContract) error {
	switch value.DataType {
	case dtype.F32, dtype.F16, dtype.BF16:
	default:
		return errors.New("representation: tensor data type is not an admitted floating activation type")
	}
	if len(value.Axes) == tensor.FirstOffset || len(value.Axes) > tensor.MaxDimensions {
		return fmt.Errorf("representation: tensor rank must be in [1,%d]", tensor.MaxDimensions)
	}
	seen := make(map[AxisKind]struct{}, len(value.Axes))
	maximumElements := uint64(tensor.SingletonExtent)
	channel := false
	for _, axis := range value.Axes {
		if !axis.Kind.valid() {
			return errors.New("representation: invalid tensor axis")
		}
		if _, duplicate := seen[axis.Kind]; duplicate {
			return errors.New("representation: duplicate tensor axis")
		}
		seen[axis.Kind] = struct{}{}
		maximum, err := axis.Bounds.maximum()
		if err != nil {
			return err
		}
		var ok bool
		maximumElements, ok = checked.Mul64(maximumElements, maximum)
		if !ok {
			return errors.New("representation: maximum tensor extent overflows")
		}
		if axis.Kind == AxisChannel {
			if !checked.Nonzero(axis.Bounds.Extent) {
				return errors.New("representation: channel width must be exact")
			}
			channel = true
		}
	}
	if !channel || !checked.Nonzero(maximumElements) {
		return errors.New("representation: tensor has no exact channel axis")
	}
	return nil
}

func (b AxisBounds) maximum() (uint64, error) {
	switch {
	case checked.Nonzero(b.Extent) && !checked.Nonzero(b.Minimum) && !checked.Nonzero(b.Maximum):
		return b.Extent, nil
	case !checked.Nonzero(b.Extent) && checked.Nonzero(b.Minimum) && b.Maximum >= b.Minimum:
		return b.Maximum, nil
	default:
		return tensor.FirstOffset, errors.New("representation: axis requires one exact extent or a valid inclusive range")
	}
}

func validateSequence(value SequenceContract, axes []Axis) error {
	if !value.Axis.valid() || value.Axis == AxisChannel || value.Axis == AxisBatch || !hasAxis(axes, value.Axis) {
		return errors.New("representation: sequence axis is absent or invalid")
	}
	switch value.Mask {
	case MaskNone:
		if value.Padding != PaddingNone {
			return errors.New("representation: unmasked sequence cannot declare padding")
		}
	case MaskPrefix:
		if value.Padding != PaddingSuffix {
			return errors.New("representation: prefix mask requires suffix padding")
		}
	case MaskExplicit:
		if value.Padding != PaddingNone && value.Padding != PaddingSuffix {
			return errors.New("representation: invalid explicit-mask padding")
		}
	default:
		return errors.New("representation: invalid mask policy")
	}
	switch value.Position {
	case PositionNone:
		if len(value.PositionAxes) != tensor.FirstOffset {
			return errors.New("representation: position-free sequence declares axes")
		}
	case PositionSequential:
		if len(value.PositionAxes) != tensor.SingletonExtent || value.PositionAxes[tensor.FirstOffset] != value.Axis {
			return errors.New("representation: sequential positions must bind the sequence axis")
		}
	case PositionMultiAxis:
		if len(value.PositionAxes) < tensor.PairedExtent {
			return errors.New("representation: multi-axis positions require multiple axes")
		}
		seen := make(map[AxisKind]struct{}, len(value.PositionAxes))
		for _, axis := range value.PositionAxes {
			if !hasAxis(axes, axis) {
				return errors.New("representation: position axis is absent from tensor layout")
			}
			if _, duplicate := seen[axis]; duplicate {
				return errors.New("representation: duplicate position axis")
			}
			seen[axis] = struct{}{}
		}
	default:
		return errors.New("representation: invalid position policy")
	}
	sort.Slice(value.SpecialTokens, func(i, j int) bool {
		return value.SpecialTokens[i].Role < value.SpecialTokens[j].Role
	})
	for index, token := range value.SpecialTokens {
		if !token.Role.valid() || token.Disposition != TokenKeep && token.Disposition != TokenDrop {
			return errors.New("representation: invalid special-token policy")
		}
		if index > tensor.FirstOffset && value.SpecialTokens[index-tensor.SingletonExtent].Role == token.Role {
			return errors.New("representation: duplicate special-token role")
		}
	}
	return nil
}

func validateNormalization(value NormalizationContract) error {
	if value.Magnitude != MagnitudeNative && value.Magnitude != MagnitudeUnit && value.Magnitude != MagnitudePreserve {
		return errors.New("representation: invalid magnitude policy")
	}
	switch value.Kind {
	case NormalizationNone:
		if checked.Nonzero(value.Epsilon) {
			return errors.New("representation: normalization-free contract has epsilon")
		}
	case NormalizationRMS, NormalizationLayer, NormalizationL2:
		if !checked.PositiveFinite32(value.Epsilon) {
			return errors.New("representation: normalized contract requires a positive finite epsilon")
		}
	default:
		return errors.New("representation: invalid normalization policy")
	}
	return nil
}

func validateAuthority(value Authority) error {
	expected := artifact.KindInvalid
	switch value.Role {
	case AuthorityTokenizer:
		expected = artifact.KindTokenizer
	case AuthorityProjector:
		expected = artifact.KindProjector
	case AuthorityProcessor, AuthorityProfile:
		expected = artifact.KindProfile
	case AuthorityAdapter:
		expected = artifact.KindAdapter
	default:
		return errors.New("representation: invalid authority role")
	}
	if value.Artifact.Kind() != expected {
		return errors.New("representation: authority artifact kind differs from role")
	}
	return nil
}

func hasAxis(axes []Axis, kind AxisKind) bool {
	return slices.ContainsFunc(axes, func(axis Axis) bool { return axis.Kind == kind })
}

func (m Modality) valid() bool {
	switch m {
	case ModalityText, ModalityImage, ModalityAudio, ModalityVideo,
		ModalityAction, ModalitySeries, ModalityTabular, ModalityLatent:
		return true
	default:
		return false
	}
}

func (a AxisKind) valid() bool {
	switch a {
	case AxisChannel, AxisSequence, AxisBatch, AxisWidth, AxisHeight, AxisTime:
		return true
	default:
		return false
	}
}

func (r SpecialTokenRole) valid() bool {
	switch r {
	case SpecialTokenBOS, SpecialTokenEOS, SpecialTokenCLS, SpecialTokenSeparator:
		return true
	default:
		return false
	}
}
