package inference

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"math"

	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/sampling"
	"overgo/internal/statecodec"
	"overgo/internal/tokenizer"
)

const (
	sessionStateMagic = "L2GSES01"
	sessionHeaderSize = 56
	maxSessionTokens  = 1 << 24
	maxSamplerState   = 1 << 20
)

// Session: resumable generation state; final token is intentionally
// pending: Cache contains every token before it, so ContinueSession can
// evaluate that token and produce next one without recomputing prompt
type Session struct {
	TokenIDs []tokenizer.TokenID
	Cache    *KVCache
}

// SaveSession: serializes token history, KV tensors, and sampler RNG/adaptive
// state; state is bound to fingerprint of loaded GGUF model
func (r *Runner) SaveSession(session *Session, sampler *sampling.Sampler) ([]byte, error) {
	if r == nil {
		return nil, errRunnerNil
	}
	if err := r.validateSession(session); err != nil {
		return nil, err
	}
	cacheData, err := r.SaveCache(session.Cache)
	if err != nil {
		return nil, err
	}
	samplerData, err := sampler.SaveState()
	if err != nil {
		return nil, fmt.Errorf("inference: save sampler state: %w", err)
	}
	signature, err := r.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	encoder := statecodec.NewEncoder(uint64(math.MaxInt))
	encoder.Raw([]byte(sessionStateMagic))
	encoder.Raw(signature[:])
	encoder.U32(uint32(len(session.TokenIDs)))
	encoder.U64(uint64(len(cacheData)))
	encoder.U32(uint32(len(samplerData)))
	for _, tokenID := range session.TokenIDs {
		encoder.U32(uint32(tokenID))
	}
	encoder.Raw(cacheData)
	encoder.Raw(samplerData)
	output, err := encoder.Data()
	if err != nil {
		return nil, errors.New("inference: session state exceeds addressable memory")
	}
	return output, nil
}

// LoadSession: parses and validates untrusted session state, then restores
// supplied sampler only after complete session has passed validation
func (r *Runner) LoadSession(data []byte, sampler *sampling.Sampler) (*Session, error) {
	if r == nil {
		return nil, errRunnerNil
	}
	if sampler == nil {
		return nil, errors.New("inference: sampler is nil")
	}
	decoder, err := r.modelStateDecoder(data, sessionStateMagic, "session")
	if err != nil {
		return nil, err
	}
	tokenCount := decoder.U32()
	cacheLength := decoder.U64()
	samplerLength := decoder.U32()
	if decoder.Err() != nil {
		return nil, errors.New("inference: session state is truncated")
	}
	if tokenCount == 0 || tokenCount > maxSessionTokens {
		return nil, errors.New("inference: session token count is invalid or exceeds limit")
	}
	if samplerLength == 0 || samplerLength > maxSamplerState {
		return nil, errors.New("inference: sampler state size is invalid or exceeds limit")
	}
	tokenBytes, _ := checked.Bytes(uint64(tokenCount), 4)
	payloadLength := decoder.Remaining()
	if tokenBytes > payloadLength ||
		cacheLength > payloadLength-tokenBytes ||
		uint64(samplerLength) != payloadLength-tokenBytes-cacheLength {
		return nil, errors.New("inference: session payload lengths are invalid")
	}

	tokens := make([]tokenizer.TokenID, int(tokenCount))
	for index := range tokens {
		raw := decoder.U32()
		if raw > math.MaxInt32 {
			return nil, fmt.Errorf("inference: session token %d is outside token ID range", index)
		}
		tokens[index] = tokenizer.TokenID(raw)
	}
	cacheData := decoder.Raw(cacheLength)
	samplerData := decoder.Raw(uint64(samplerLength))
	if decoder.Done() != nil {
		return nil, errors.New("inference: session payload lengths are invalid")
	}
	cache, err := r.LoadCache(cacheData)
	if err != nil {
		return nil, fmt.Errorf("inference: load session cache: %w", err)
	}
	session := &Session{TokenIDs: tokens, Cache: cache}
	if err := r.validateSession(session); err != nil {
		return nil, err
	}
	if err := sampler.LoadState(samplerData); err != nil {
		return nil, fmt.Errorf("inference: load sampler state: %w", err)
	}
	return session, nil
}

func (r *Runner) modelStateDecoder(data []byte, magic, label string) (*statecodec.Decoder, error) {
	if r == nil {
		return nil, errRunnerNil
	}
	decoder := statecodec.NewDecoder(data, uint64(math.MaxInt))
	encodedMagic := decoder.Raw(uint64(len(magic)))
	modelSignature := decoder.Raw(sha256.Size)
	if decoder.Err() != nil {
		return nil, fmt.Errorf("inference: %s state is truncated", label)
	}
	if !bytes.Equal(encodedMagic, []byte(magic)) {
		return nil, fmt.Errorf("inference: %s state has invalid magic or version", label)
	}
	signature, err := r.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if !bytes.Equal(modelSignature, signature[:]) {
		return nil, fmt.Errorf("inference: %s state belongs to a different model", label)
	}
	return decoder, nil
}

func (r *Runner) validateSession(session *Session) error {
	if session == nil {
		return errors.New("inference: session is nil")
	}
	if len(session.TokenIDs) == 0 || len(session.TokenIDs) > maxSessionTokens {
		return errors.New("inference: session token count is invalid or exceeds limit")
	}
	if err := r.validateCache(session.Cache); err != nil {
		return err
	}
	if uint64(effectiveCachePosition(session.Cache))+1 != uint64(len(session.TokenIDs)) {
		return fmt.Errorf(
			"inference: session has %d tokens but cache next position is %d; need exactly one pending token",
			len(session.TokenIDs),
			effectiveCachePosition(session.Cache),
		)
	}
	if r.spec.ContextLength > 0 && session.Cache.Tokens > r.spec.ContextLength {
		return fmt.Errorf(
			"inference: session cache token count %d exceeds context state limit %d",
			session.Cache.Tokens,
			r.spec.ContextLength,
		)
	}
	vocabularySize := r.spec.VocabularySize
	if vocabularySize == 0 && r.vocab != nil {
		vocabularySize = uint32(len(r.vocab.Tokens))
	}
	for index, tokenID := range session.TokenIDs {
		if tokenID < 0 || (vocabularySize > 0 && uint32(tokenID) >= vocabularySize) {
			return fmt.Errorf("inference: session token %d has invalid ID %d", index, tokenID)
		}
	}
	return nil
}

func (r *Runner) sessionModelSignature() ([32]byte, error) {
	r.modelSignatureOnce.Do(func() {
		hasher := sha256.New()
		writeFingerprintString(hasher, r.spec.Architecture)
		writeFingerprintString(hasher, r.spec.Name)
		for _, value := range []uint32{
			r.spec.BlockCount,
			r.spec.DecoderBlockCount,
			r.spec.ContextLength,
			r.spec.EmbeddingLength,
			r.spec.FeedForwardLength,
			r.spec.HeadCount,
			r.spec.HeadCountKV,
			r.spec.KeyLength,
			r.spec.ValueLength,
			r.spec.KeyLengthSWA,
			r.spec.ValueLengthSWA,
			math.Float32bits(r.spec.RopeFrequencyBase),
			math.Float32bits(r.spec.RMSNormEpsilon),
			math.Float32bits(r.spec.LayerNormEpsilon),
			r.spec.VocabularySize,
			r.spec.RopeDimensionCount,
			r.spec.SSMConvKernel,
			r.spec.SSMInnerSize,
			r.spec.SSMStateSize,
			r.spec.SSMTimeStepRank,
			r.spec.SSMGroupCount,
			r.spec.FullAttentionInterval,
			r.spec.EmbeddingPerLayer,
			r.spec.SharedKVLayers,
			r.spec.RelativeBuckets,
			r.spec.DecoderStartTokenID,
		} {
			var encoded [4]byte
			binary.LittleEndian.PutUint32(encoded[:], value)
			_, _ = hasher.Write(encoded[:])
		}
		for _, section := range r.spec.RopeSections {
			var encoded [4]byte
			binary.LittleEndian.PutUint32(encoded[:], uint32(section))
			_, _ = hasher.Write(encoded[:])
		}
		for _, heads := range r.spec.LayerHeadCounts {
			var encoded [4]byte
			binary.LittleEndian.PutUint32(encoded[:], heads)
			_, _ = hasher.Write(encoded[:])
		}
		for _, heads := range r.spec.LayerKVHeadCounts {
			var encoded [4]byte
			binary.LittleEndian.PutUint32(encoded[:], heads)
			_, _ = hasher.Write(encoded[:])
		}
		for _, width := range r.spec.LayerFeedForward {
			var encoded [4]byte
			binary.LittleEndian.PutUint32(encoded[:], width)
			_, _ = hasher.Write(encoded[:])
		}
		for _, sliding := range r.spec.SlidingLayers {
			if sliding {
				_, _ = hasher.Write([]byte{1})
			} else {
				_, _ = hasher.Write([]byte{0})
			}
		}
		for _, recurrent := range r.spec.RecurrentLayers {
			if recurrent {
				_, _ = hasher.Write([]byte{1})
			} else {
				_, _ = hasher.Write([]byte{0})
			}
		}
		if r.file != nil {
			for _, metadata := range r.file.Metadata {
				writeFingerprintString(hasher, metadata.Key)
				writeFingerprintString(hasher, fmt.Sprintf(
					"%d/%d/%#v",
					metadata.Value.Type,
					metadata.Value.ArrayType,
					metadata.Value.Data,
				))
			}
			for _, info := range r.file.Tensors {
				writeFingerprintString(hasher, info.Name)
				var encoded [8]byte
				for _, value := range []uint64{
					uint64(info.Dimensions),
					info.Shape[0],
					info.Shape[1],
					info.Shape[2],
					info.Shape[3],
					uint64(info.Type),
					info.Offset,
					info.Size,
				} {
					binary.LittleEndian.PutUint64(encoded[:], value)
					_, _ = hasher.Write(encoded[:])
				}
				if err := hashTensorEdges(hasher, r.file, info); err != nil {
					r.modelSignatureErr = fmt.Errorf(
						"inference: fingerprint tensor %q: %w",
						info.Name,
						err,
					)
					return
				}
			}
		}
		copy(r.modelSignature[:], hasher.Sum(nil))
	})
	if r.modelSignatureErr != nil || len(r.loraAdapters) == 0 {
		return r.modelSignature, r.modelSignatureErr
	}
	hasher := sha256.New()
	_, _ = hasher.Write(r.modelSignature[:])
	loraSignature := r.currentLoRASignature()
	_, _ = hasher.Write(loraSignature[:])
	var result [32]byte
	copy(result[:], hasher.Sum(nil))
	return result, nil
}

func writeFingerprintString(hasher hash.Hash, value string) {
	var length [8]byte
	binary.LittleEndian.PutUint64(length[:], uint64(len(value)))
	_, _ = hasher.Write(length[:])
	_, _ = hasher.Write([]byte(value))
}

func hashTensorEdges(
	hasher hash.Hash,
	file *gguf.File,
	info gguf.TensorInfo,
) error {
	const sampleSize = uint64(64)
	if info.Size == 0 {
		return nil
	}
	length := min(info.Size, sampleSize)
	data := make([]byte, int(length))
	if err := file.ReadTensorRange(info, 0, data); err != nil {
		return err
	}
	_, _ = hasher.Write(data)
	if info.Size > length {
		if err := file.ReadTensorRange(info, info.Size-length, data); err != nil {
			return err
		}
		_, _ = hasher.Write(data)
	}
	return nil
}
