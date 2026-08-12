// Incremental decode: O(n) per-token stepping over a per-layer KV cache
// plus static cross-attention memory. Every per-row operation matches the
// full-recompute path exactly (hostmath.CausalAttentionStep is bit-identical
// to the full core's row), so stepped logits equal DecodeFull's bit-for-bit.
package seq2seq

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
)

// Decoder: one incremental decode session bound to an encoded memory.
type Decoder struct {
	m                *Model
	cross            *crossMemory
	kCache, vCache   [][]float32 // per self layer, [maxRows][kvHeads][headDim]
	pos, maxRows     int
	row, normed      []float32
	q, kRow, vRow    []float32
	attn, oProjected []float32
}

// NewDecoder prepares an incremental session over an encoded memory with
// capacity for maxRows target tokens.
func (m *Model) NewDecoder(memory []float32, memRows, maxRows int) (*Decoder, error) {
	if maxRows <= 0 {
		return nil, fmt.Errorf("seq2seq: decode capacity %d", maxRows)
	}
	cross, err := m.projectCrossMemory(memory, memRows)
	if err != nil {
		return nil, err
	}
	dims := m.Dims
	kvWidth := dims.KVHeads * dims.HeadDim
	session := &Decoder{
		m: m, cross: cross, maxRows: maxRows,
		row:    make([]float32, dims.DModel),
		normed: make([]float32, dims.DModel),
		q:      make([]float32, dims.Heads*dims.HeadDim),
		kRow:   make([]float32, kvWidth),
		vRow:   make([]float32, kvWidth),
		attn:   make([]float32, dims.Heads*dims.HeadDim),
	}
	for range m.decoderSelf {
		session.kCache = append(session.kCache, make([]float32, maxRows*kvWidth))
		session.vCache = append(session.vCache, make([]float32, maxRows*kvWidth))
	}
	return session, nil
}

// Advance consumes one target token; a non-nil logits destination receives
// the next-token logits ([vocab]).
func (s *Decoder) Advance(logits []float32, token int) error {
	if s.pos >= s.maxRows {
		return fmt.Errorf("seq2seq: decode position %d at capacity %d", s.pos, s.maxRows)
	}
	if logits != nil && len(logits) < s.m.Dims.Vocab {
		return fmt.Errorf("seq2seq: logits destination %d, want %d", len(logits), s.m.Dims.Vocab)
	}
	m, dims := s.m, s.m.Dims
	if err := m.embedRows(s.row, []int{token}); err != nil {
		return err
	}
	kvWidth := dims.KVHeads * dims.HeadDim
	for layer := range m.decoderSelf {
		self := &m.decoderSelf[layer]
		hostmath.RMSNormInto(s.normed, s.row, self.inNorm, 1, dims.DModel, dims.RMSEps)
		m.projectQ(s.q, s.normed, 1, self, s.pos, true)
		m.projectKV(s.kRow, s.vRow, s.normed, 1, self, s.pos, true)
		copy(s.kCache[layer][s.pos*kvWidth:], s.kRow)
		copy(s.vCache[layer][s.pos*kvWidth:], s.vRow)
		cached := (s.pos + 1) * kvWidth
		hostmath.CausalAttentionStep(s.attn, s.q, s.kCache[layer][:cached], s.vCache[layer][:cached], s.pos+1, dims.Heads, dims.KVHeads, dims.HeadDim)
		m.gatedResidualOut(s.row, s.attn, self.o, 1, self.gate)

		crossBlock := &m.decoderCross[layer]
		hostmath.RMSNormInto(s.normed, s.row, crossBlock.inNorm, 1, dims.DModel, dims.RMSEps)
		m.projectQ(s.q, s.normed, 1, crossBlock, 0, false)
		hostmath.MaskedBidirectionalAttention(s.attn, s.q, s.cross.k[layer], s.cross.v[layer], 1, s.cross.rows, dims.Heads, dims.KVHeads, dims.HeadDim, nil)
		m.gatedResidualOut(s.row, s.attn, crossBlock.o, 1, crossBlock.gate)
	}
	s.pos++
	if logits != nil {
		m.projectLogits(logits, s.row, s.normed)
	}
	return nil
}

type encodedRequest struct {
	memory     []float32
	sourceRows int
	maxTokens  int
}

type generationSession struct {
	decoder   *Decoder
	maxTokens int
}

type tokenSelector interface {
	selectTokens() ([]int, error)
}

func (m *Model) encodeRequest(request GenerateRequest) (encodedRequest, error) {
	if err := ValidateGenerateRequest(request); err != nil {
		return encodedRequest{}, err
	}
	memory, err := m.Encode(request.Source)
	if err != nil {
		return encodedRequest{}, err
	}
	return encodedRequest{memory: memory, sourceRows: len(request.Source), maxTokens: request.MaxTokens}, nil
}

func (m *Model) prepareGeneration(encoded encodedRequest) (tokenSelector, error) {
	if encoded.sourceRows <= 0 || encoded.maxTokens <= 0 {
		return nil, fmt.Errorf("seq2seq: invalid encoded request")
	}
	decoder, err := m.NewDecoder(encoded.memory, encoded.sourceRows, encoded.maxTokens+1)
	if err != nil {
		return nil, err
	}
	return &generationSession{decoder: decoder, maxTokens: encoded.maxTokens}, nil
}

func (s *generationSession) selectTokens() ([]int, error) {
	if s == nil || s.decoder == nil || s.maxTokens <= 0 {
		return nil, fmt.Errorf("seq2seq: invalid generation session")
	}
	model := s.decoder.m
	logits := make([]float32, model.Dims.Vocab)
	if err := s.decoder.Advance(logits, model.Dims.StartToken); err != nil {
		return nil, err
	}
	tokens := make([]int, 0, s.maxTokens)
	for len(tokens) < s.maxTokens {
		next := argmax(logits)
		if next == model.Dims.EOSToken {
			break
		}
		tokens = append(tokens, next)
		if len(tokens) == s.maxTokens {
			break
		}
		if err := s.decoder.Advance(logits, next); err != nil {
			return tokens, err
		}
	}
	return tokens, nil
}

func argmax(values []float32) int {
	best, bestAt := float32(math.Inf(-1)), 0
	for i, value := range values {
		if value > best {
			best, bestAt = value, i
		}
	}
	return bestAt
}
