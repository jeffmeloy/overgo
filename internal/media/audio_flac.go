package media

import (
	"bufio"
	"bytes"
	"context"
	"crypto/md5"
	"errors"
	"fmt"
	"io"
	"math"

	"github.com/mewkiz/flac"

	"overgo/internal/recipecontract"
)

const flacMinimumSampleBits = 4

type audioReadFunc func([]byte) (int, error)

// Read delegates byte reads to the context-checked source function.
func (read audioReadFunc) Read(data []byte) (int, error) { return read(data) }

func decodeFLAC(ctx context.Context, data []byte, maximumSamples uint64) (DecodedAudio, error) {
	source := bytes.NewReader(data)
	reader := bufio.NewReader(audioReadFunc(func(buffer []byte) (int, error) {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		return source.Read(buffer)
	}))
	// flac.New reuses this bufio.Reader. Peeking before each frame prevents an
	// incomplete sync word from being mistaken for a graceful end of stream.
	stream, err := flac.New(reader)
	if err != nil {
		return DecodedAudio{}, err
	}
	info := stream.Info
	if info.BitsPerSample < flacMinimumSampleBits || info.BitsPerSample > pcm24Bits {
		return DecodedAudio{}, fmt.Errorf("%w: FLAC %d-bit samples; supported range is %d..%d", errAudioUnsupported, info.BitsPerSample, flacMinimumSampleBits, pcm24Bits)
	}
	if info.BlockSizeMin > info.BlockSizeMax {
		return DecodedAudio{}, errors.New("media: inverted FLAC block sizes")
	}
	channels := uint64(info.NChannels)
	if info.NSamples > maximumSamples/channels {
		return DecodedAudio{}, errAudioLimit
	}
	audio := DecodedAudio{Format: recipecontract.AudioFormat{
		SampleRate: uint64(info.SampleRate), Channels: uint32(channels), Encoding: "pcm-f32le",
	}}
	if info.NSamples != 0 {
		audio.Samples = make([]float32, 0, int(info.NSamples*channels))
	}
	// MD5 is mandated by FLAC for decoded PCM integrity, not used for identity
	// or authentication. Overgo artifact identity remains SHA-256.
	digest := md5.New()
	var frames, positions uint64
	var fixed, shortBlock bool
	var fixedSize uint16
	for {
		if err := ctx.Err(); err != nil {
			return DecodedAudio{}, err
		}
		if _, err := reader.Peek(1); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return DecodedAudio{}, err
		}
		block, err := stream.Next()
		if err != nil {
			if errors.Is(err, io.EOF) {
				err = io.ErrUnexpectedEOF
			}
			return DecodedAudio{}, err
		}
		if block.BitsPerSample == 0 {
			block.BitsPerSample = info.BitsPerSample
		}
		if block.SampleRate == 0 {
			block.SampleRate = info.SampleRate
		}
		if block.BitsPerSample != info.BitsPerSample || block.SampleRate != info.SampleRate ||
			block.Channels.Count() != int(channels) || block.BlockSize == 0 || block.BlockSize > info.BlockSizeMax {
			return DecodedAudio{}, errors.New("media: FLAC frame geometry differs from STREAMINFO")
		}
		if frames == 0 {
			fixed, fixedSize = block.HasFixedBlockSize, block.BlockSize
		}
		if block.HasFixedBlockSize != fixed || fixed && (block.Num != frames || shortBlock || block.BlockSize > fixedSize) ||
			!fixed && block.Num != positions {
			return DecodedAudio{}, errors.New("media: discontinuous FLAC frame sequence")
		}
		shortBlock = fixed && block.BlockSize < fixedSize
		next := positions + uint64(block.BlockSize)
		if next > maximumSamples/channels {
			return DecodedAudio{}, errAudioLimit
		}
		if info.NSamples != 0 && next > info.NSamples {
			return DecodedAudio{}, errors.New("media: FLAC frame exceeds declared sample count")
		}
		if err := block.Parse(); err != nil {
			return DecodedAudio{}, err
		}
		magnitude := int32(1) << (info.BitsPerSample - 1)
		for _, channel := range block.Subframes {
			if len(channel.Samples) != int(block.BlockSize) {
				return DecodedAudio{}, errors.New("media: FLAC subframe sample count differs")
			}
			for _, sample := range channel.Samples {
				if sample < -magnitude || sample >= magnitude {
					return DecodedAudio{}, errors.New("media: FLAC sample exceeds declared bit depth")
				}
			}
		}
		block.Hash(digest)
		for index := range int(block.BlockSize) {
			for _, channel := range block.Subframes {
				audio.Samples = append(audio.Samples, float32(math.Ldexp(float64(channel.Samples[index]), 1-int(info.BitsPerSample))))
			}
		}
		positions, frames = next, frames+1
	}
	if frames == 0 || info.NSamples != 0 && positions != info.NSamples {
		return DecodedAudio{}, io.ErrUnexpectedEOF
	}
	if info.MD5sum != [md5.Size]byte{} && !bytes.Equal(info.MD5sum[:], digest.Sum(nil)) {
		return DecodedAudio{}, errors.New("media: FLAC decoded PCM checksum differs")
	}
	return audio, nil
}
