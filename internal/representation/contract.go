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
	ContractVersion   uint16 = artifact.InitialDocumentVersion
	ContractMediaType        = "application/vnd.overgo.representation-contract+json"
	ContractSchema           = "overgo/representation-contract/v1"
)

type TapPoint string

const (
	TapEmbeddingOutput TapPoint = "embedding-output"
	TapEncoderOutput   TapPoint = "encoder-output"
	TapLayerInput      TapPoint = "layer-input"
	TapLayerOutput     TapPoint = "layer-output"
	TapAttentionInput  TapPoint = "attention-input"
)

type Modality string

const (
	ModalityText    Modality = "text"
	ModalityImage   Modality = "image"
	ModalityAudio   Modality = "audio"
	ModalityVideo   Modality = "video"
	ModalityAction  Modality = "action"
	ModalitySeries  Modality = "series"
	ModalityTabular Modality = "tabular"
	ModalityLatent  Modality = "latent"
)

type AxisKind string

const (
	AxisChannel  AxisKind = "channel"
	AxisSequence AxisKind = "sequence"
	AxisBatch    AxisKind = "batch"
	AxisWidth    AxisKind = "width"
	AxisHeight   AxisKind = "height"
	AxisTime     AxisKind = "time"
)

// AxisBounds is either one exact Extent or an inclusive dynamic range.
type AxisBounds struct {
	Extent  uint64 `json:"extent,omitempty"`
	Minimum uint64 `json:"minimum,omitempty"`
	Maximum uint64 `json:"maximum,omitempty"`
}

type Axis struct {
	Kind   AxisKind   `json:"kind"`
	Bounds AxisBounds `json:"bounds"`
}

type TensorContract struct {
	DataType dtype.Type `json:"data_type"`
	Axes     []Axis     `json:"axes"`
}

type MaskPolicy string

const (
	MaskNone     MaskPolicy = "none"
	MaskPrefix   MaskPolicy = "valid-prefix"
	MaskExplicit MaskPolicy = "explicit"
)

type PaddingPolicy string

const (
	PaddingNone   PaddingPolicy = "none"
	PaddingSuffix PaddingPolicy = "suffix"
)

type PositionPolicy string

const (
	PositionNone       PositionPolicy = "none"
	PositionSequential PositionPolicy = "sequential"
	PositionMultiAxis  PositionPolicy = "multi-axis"
)

type SpecialTokenRole string

const (
	SpecialTokenBOS       SpecialTokenRole = "bos"
	SpecialTokenEOS       SpecialTokenRole = "eos"
	SpecialTokenCLS       SpecialTokenRole = "cls"
	SpecialTokenSeparator SpecialTokenRole = "separator"
)

type TokenDisposition string

const (
	TokenKeep TokenDisposition = "keep"
	TokenDrop TokenDisposition = "drop"
)

type SpecialToken struct {
	Role        SpecialTokenRole `json:"role"`
	Disposition TokenDisposition `json:"disposition"`
}

type SequenceContract struct {
	Axis          AxisKind       `json:"axis"`
	Mask          MaskPolicy     `json:"mask"`
	Padding       PaddingPolicy  `json:"padding"`
	Position      PositionPolicy `json:"position"`
	PositionAxes  []AxisKind     `json:"position_axes,omitempty"`
	SpecialTokens []SpecialToken `json:"special_tokens,omitempty"`
}

type NormalizationKind string

const (
	NormalizationNone  NormalizationKind = "none"
	NormalizationRMS   NormalizationKind = "rms"
	NormalizationLayer NormalizationKind = "layer"
	NormalizationL2    NormalizationKind = "l2"
)

type MagnitudePolicy string

const (
	MagnitudeNative   MagnitudePolicy = "native"
	MagnitudeUnit     MagnitudePolicy = "unit"
	MagnitudePreserve MagnitudePolicy = "preserve-source"
)

type NormalizationContract struct {
	Kind      NormalizationKind `json:"kind"`
	Epsilon   float32           `json:"epsilon,omitempty"`
	Magnitude MagnitudePolicy   `json:"magnitude"`
}

type AuthorityRole string

const (
	AuthorityTokenizer AuthorityRole = "tokenizer"
	AuthorityProjector AuthorityRole = "projector"
	AuthorityProcessor AuthorityRole = "processor"
	AuthorityProfile   AuthorityRole = "profile"
	AuthorityAdapter   AuthorityRole = "adapter"
)

type Authority struct {
	Role     AuthorityRole `json:"role"`
	Artifact artifact.ID   `json:"artifact"`
}

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

func (c Contract) ValidateIdentity() error { return contractCodec.ValidateIdentity(c) }

func (c Contract) Content() (artifact.Content, error) { return contractCodec.Content(c) }

func (c Contract) Lineage() []artifact.Lineage {
	parents := []artifact.ID{c.Producer.Model, c.Producer.Definition}
	for _, authority := range c.Authorities {
		parents = append(parents, authority.Artifact)
	}
	return artifact.DependencyLineage(c.ID, parents...)
}

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
