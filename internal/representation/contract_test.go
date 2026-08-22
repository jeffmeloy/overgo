package representation

import (
	"encoding/json"
	"math"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/tensor/dtype"
)

func TestRepresentationContract(t *testing.T) {
	model := contractTestID(t, artifact.KindModel, "encoder")
	definition := contractTestID(t, artifact.KindModelDefinition, "encoder-definition")
	tokenizer := contractTestID(t, artifact.KindTokenizer, "encoder-tokenizer")
	processor := contractTestID(t, artifact.KindProfile, "audio-processor")
	base := func() (Producer, TensorContract, SequenceContract, NormalizationContract, []Authority) {
		layer := uint32(7)
		return Producer{Model: model, Definition: definition, Tap: TapLayerOutput, Layer: &layer},
			TensorContract{DataType: dtype.F32, Axes: []Axis{
				{Kind: AxisChannel, Bounds: AxisBounds{Extent: 768}},
				{Kind: AxisTime, Bounds: AxisBounds{Minimum: 1, Maximum: 2048}},
				{Kind: AxisBatch, Bounds: AxisBounds{Extent: 1}},
			}},
			SequenceContract{
				Axis: AxisTime, Mask: MaskPrefix, Padding: PaddingSuffix,
				Position: PositionSequential, PositionAxes: []AxisKind{AxisTime},
				SpecialTokens: []SpecialToken{
					{Role: SpecialTokenEOS, Disposition: TokenDrop},
					{Role: SpecialTokenBOS, Disposition: TokenKeep},
				},
			},
			NormalizationContract{Kind: NormalizationRMS, Epsilon: 1e-5, Magnitude: MagnitudePreserve},
			[]Authority{
				{Role: AuthorityProcessor, Artifact: processor},
				{Role: AuthorityTokenizer, Artifact: tokenizer},
			}
	}

	producer, tensorContract, sequence, normalization, authorities := base()
	contract, err := contractCodec.New(Contract{
		Version: ContractVersion, Producer: producer, Modality: ModalityAudio,
		Tensor: tensorContract, Sequence: sequence, Normalization: normalization,
		Authorities: authorities,
	})
	if err != nil {
		t.Fatal(err)
	}
	if contract.ID.Kind() != artifact.KindProfile || contract.Version != ContractVersion {
		t.Fatalf("identity/version = %s/%d", contract.ID, contract.Version)
	}
	if contract.Authorities[0].Role != AuthorityProcessor || contract.Sequence.SpecialTokens[0].Role != SpecialTokenBOS {
		t.Fatalf("contract is not canonical: %+v", contract)
	}
	if err := contract.ValidateIdentity(); err != nil {
		t.Fatal(err)
	}
	content, err := contract.Content()
	if err != nil {
		t.Fatal(err)
	}
	if content.Descriptor.ID != contract.ID || content.Descriptor.MediaType != ContractMediaType || content.Descriptor.Schema != ContractSchema {
		t.Fatalf("descriptor = %+v", content.Descriptor)
	}
	parsed, err := contractCodec.Parse(content.Data)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.ID != contract.ID || !slices.Equal(parsed.Lineage(), contract.Lineage()) {
		t.Fatalf("round trip identity/lineage differs")
	}

	reordered := slices.Clone(authorities)
	slices.Reverse(reordered)
	same, err := contractCodec.New(Contract{
		Version: ContractVersion, Producer: producer, Modality: ModalityAudio,
		Tensor: tensorContract, Sequence: sequence, Normalization: normalization,
		Authorities: reordered,
	})
	if err != nil || same.ID != contract.ID {
		t.Fatalf("canonical identity = %s, %v; want %s", same.ID, err, contract.ID)
	}

	t.Run("rejects incompatible authority and layout facts", func(t *testing.T) {
		cases := []struct {
			name string
			edit func(*Producer, *TensorContract, *SequenceContract, *NormalizationContract, *[]Authority)
		}{
			{"model kind", func(p *Producer, _ *TensorContract, _ *SequenceContract, _ *NormalizationContract, _ *[]Authority) {
				p.Model = tokenizer
			}},
			{"definition kind", func(p *Producer, _ *TensorContract, _ *SequenceContract, _ *NormalizationContract, _ *[]Authority) {
				p.Definition = model
			}},
			{"boundary layer", func(p *Producer, _ *TensorContract, _ *SequenceContract, _ *NormalizationContract, _ *[]Authority) {
				p.Tap = TapEncoderOutput
			}},
			{"missing layer", func(p *Producer, _ *TensorContract, _ *SequenceContract, _ *NormalizationContract, _ *[]Authority) {
				p.Layer = nil
			}},
			{"quantized activation", func(_ *Producer, c *TensorContract, _ *SequenceContract, _ *NormalizationContract, _ *[]Authority) {
				c.DataType = dtype.Q4_0
			}},
			{"duplicate axis", func(_ *Producer, c *TensorContract, _ *SequenceContract, _ *NormalizationContract, _ *[]Authority) {
				c.Axes[2].Kind = AxisTime
			}},
			{"dynamic channel", func(_ *Producer, c *TensorContract, _ *SequenceContract, _ *NormalizationContract, _ *[]Authority) {
				c.Axes[0].Bounds = AxisBounds{Minimum: 1, Maximum: 768}
			}},
			{"invalid bounds", func(_ *Producer, c *TensorContract, _ *SequenceContract, _ *NormalizationContract, _ *[]Authority) {
				c.Axes[1].Bounds = AxisBounds{Minimum: 2, Maximum: 1}
			}},
			{"extent overflow", func(_ *Producer, c *TensorContract, _ *SequenceContract, _ *NormalizationContract, _ *[]Authority) {
				c.Axes[1].Bounds = AxisBounds{Extent: math.MaxUint64}
			}},
			{"missing sequence axis", func(_ *Producer, _ *TensorContract, s *SequenceContract, _ *NormalizationContract, _ *[]Authority) {
				s.Axis = AxisWidth
			}},
			{"padding without mask", func(_ *Producer, _ *TensorContract, s *SequenceContract, _ *NormalizationContract, _ *[]Authority) {
				s.Mask = MaskNone
			}},
			{"position mismatch", func(_ *Producer, _ *TensorContract, s *SequenceContract, _ *NormalizationContract, _ *[]Authority) {
				s.PositionAxes = []AxisKind{AxisBatch}
			}},
			{"duplicate special token", func(_ *Producer, _ *TensorContract, s *SequenceContract, _ *NormalizationContract, _ *[]Authority) {
				s.SpecialTokens[1].Role = SpecialTokenEOS
			}},
			{"nonfinite epsilon", func(_ *Producer, _ *TensorContract, _ *SequenceContract, n *NormalizationContract, _ *[]Authority) {
				n.Epsilon = float32(math.Inf(1))
			}},
			{"authority kind", func(_ *Producer, _ *TensorContract, _ *SequenceContract, _ *NormalizationContract, a *[]Authority) {
				(*a)[0].Artifact = tokenizer
			}},
			{"duplicate authority", func(_ *Producer, _ *TensorContract, _ *SequenceContract, _ *NormalizationContract, a *[]Authority) {
				(*a)[0].Role = AuthorityTokenizer
			}},
		}
		for _, test := range cases {
			t.Run(test.name, func(t *testing.T) {
				producer, tensorContract, sequence, normalization, authorities := base()
				test.edit(&producer, &tensorContract, &sequence, &normalization, &authorities)
				if _, err := contractCodec.New(Contract{
					Version: ContractVersion, Producer: producer, Modality: ModalityAudio,
					Tensor: tensorContract, Sequence: sequence, Normalization: normalization,
					Authorities: authorities,
				}); err == nil {
					t.Fatal("invalid contract accepted")
				}
			})
		}
	})

	t.Run("rejects noncanonical and unknown JSON", func(t *testing.T) {
		var body map[string]any
		if err := json.Unmarshal(content.Data, &body); err != nil {
			t.Fatal(err)
		}
		body["unknown"] = true
		data, err := json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := contractCodec.Parse(data); err == nil {
			t.Fatal("unknown field accepted")
		}
	})
}

func contractTestID(t *testing.T, kind artifact.Kind, value string) artifact.ID {
	t.Helper()
	id, err := artifact.IdentifyBytes(kind, []byte(value))
	if err != nil {
		t.Fatal(err)
	}
	return id
}
