package inference

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"hash"
	"math"

	"overgo/internal/gguf"
	"overgo/internal/statecodec"
)

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
