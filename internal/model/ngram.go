package model

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
)

// NGramSpec holds the exact file-declared hash and lexical injection contract.
type NGramSpec struct {
	Layer, Size, HeadsPerOrder, RowWidth, ConvKernel, EOS, ImageToken uint32
	Multipliers, Offsets, VocabSizes                                  []uint64
}

// maxNGramToken preserves integer token history in the F32 cache representation.
const maxNGramToken = uint32(1 << 24)

func (m specMetadata) readNGram(spec *Spec) error {
	key := m.prefix + "ple.layers"
	if _, present := m.values[key]; !present {
		return nil
	}
	layers, err := requiredLayerUint32Compatible(m.values, key, tensor.SingletonExtent)
	if err != nil {
		return err
	}
	n := &NGramSpec{Layer: layers[0]}
	for _, field := range []struct {
		key string
		out *uint32
	}{
		{"ple.ngram_size", &n.Size}, {"ple.heads_per_ngram", &n.HeadsPerOrder},
		{"ple.conv_kernel", &n.ConvKernel}, {"ple.eos_token_id", &n.EOS},
		{"embedding_length_per_layer_input", &n.RowWidth},
	} {
		value, err := required[uint32](m.values, m.prefix+field.key, gguf.ValueTypeUint32)
		if err != nil {
			return err
		}
		*field.out = value
	}
	n.ImageToken = optionalOr(m.values, m.prefix+"ple.image_token_id", gguf.ValueTypeUint32, n.EOS)
	for _, field := range []struct {
		key string
		out *[]uint64
	}{
		{"ple.layer_multipliers", &n.Multipliers}, {"ple.head_offsets", &n.Offsets}, {"ple.head_vocab_sizes", &n.VocabSizes},
	} {
		value, err := requiredArray[uint64](m.values, m.prefix+field.key, gguf.ValueTypeUint64)
		if err != nil {
			return err
		}
		*field.out = value
	}
	spec.NGram = n
	return nil
}

func (s Spec) validateNGram() error {
	n := s.NGram
	if n == nil {
		return nil
	}
	if s.Profile().LayerTopology != layerTopologyGatedHyper || n.Layer >= s.BlockCount ||
		n.Size < tensor.PairedExtent || n.HeadsPerOrder == 0 || n.RowWidth == 0 || n.ConvKernel < tensor.PairedExtent ||
		s.VocabularySize == 0 || s.VocabularySize-1 > maxNGramToken || n.EOS >= s.VocabularySize || n.ImageToken >= s.VocabularySize {
		return errors.New("ngram metadata has incompatible dimensions, layer or token range")
	}
	heads := uint64(n.Size-1) * uint64(n.HeadsPerOrder)
	if uint64(len(n.Multipliers)) != uint64(n.Size) || uint64(len(n.Offsets)) != heads || uint64(len(n.VocabSizes)) != heads ||
		heads*uint64(n.RowWidth) != uint64(s.EmbeddingLength) || uint64(n.ConvKernel-1)*uint64(n.Size) > math.MaxUint32 {
		return errors.New("ngram metadata arrays or embedding width are incompatible")
	}
	for _, multiplier := range n.Multipliers {
		if multiplier == 0 || multiplier > math.MaxInt64/uint64(s.VocabularySize) {
			return errors.New("ngram multiplier does not preserve the signed int64 reference product")
		}
	}
	var end uint64
	for i, offset := range n.Offsets {
		if offset != end || n.VocabSizes[i] == 0 || n.VocabSizes[i] > math.MaxUint32-offset {
			return fmt.Errorf("ngram head %d has an invalid table range", i)
		}
		end = offset + n.VocabSizes[i]
	}
	return nil
}

// NGramRows maps each token's causal history to ordered embedding-table rows.
// Prior history is oldest first and remains unchanged on failure.
func (s Spec) NGramRows(tokens, previous []uint32) (rows, next []uint32, err error) {
	if s.NGram == nil {
		return nil, nil, errors.New("ngram metadata is unavailable")
	}
	if err := s.validateNGram(); err != nil {
		return nil, nil, err
	}
	n := s.NGram
	if len(tokens) == 0 || len(previous) != 0 && len(previous) != int(n.Size-1) ||
		len(tokens) > math.MaxInt/len(n.Offsets) {
		return nil, nil, errors.New("ngram token or history count is invalid")
	}
	history := slices.Clone(previous)
	if len(history) == 0 {
		history = slices.Repeat([]uint32{n.EOS}, int(n.Size-1))
	}
	for _, token := range history {
		if token >= s.VocabularySize {
			return nil, nil, errors.New("ngram history token exceeds vocabulary")
		}
	}
	rows = make([]uint32, len(tokens)*len(n.Offsets))
	for index, token := range tokens {
		if token >= s.VocabularySize {
			return nil, nil, fmt.Errorf("ngram token %d exceeds vocabulary", token)
		}
		mixed := uint64(token) * n.Multipliers[0]
		cut := false
		for position := uint32(1); position < n.Size; position++ {
			prior := history[len(history)-int(position)]
			cut = cut || prior == n.EOS
			if cut {
				prior = n.EOS
			}
			mixed ^= uint64(prior) * n.Multipliers[position]
			for head := uint32(0); head < n.HeadsPerOrder; head++ {
				h := (position-1)*n.HeadsPerOrder + head
				rows[index*len(n.Offsets)+int(h)] = uint32(mixed%n.VocabSizes[h] + n.Offsets[h])
			}
		}
		copy(history, history[1:])
		history[len(history)-1] = token
	}
	return rows, history, nil
}
