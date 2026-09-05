package media

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"math/bits"

	"overgo/internal/recipecontract"
)

// Codec tags and expansion follow audio.cpp wav_reader.cpp at
// 3497b7cc44753e2c141d8fe60ac42cec433e3281 (Apache-2.0; ShugoAI LLC,
// copyright 2026; see licenses/audio-cpp-Apache-2.0.txt). Container sizes,
// duplicate chunks, and frame geometry are checked more strictly here.
const (
	wavALaw            = 6
	wavMuLaw           = 7
	wavExtensible      = 0xfffe
	wavExtensibleBytes = 40
	wavExtensionBytes  = 22
	pcm24Bits          = 3 * bitsPerByte
	pcm32Bits          = 4 * bitsPerByte
	g711MantissaBits   = 4
	g711Sign           = 1 << (bitsPerByte - 1)
	g711ExponentMask   = (g711Sign - 1) >> g711MantissaBits
	g711MantissaMask   = (1 << g711MantissaBits) - 1
	g711MuBias         = 0x84
	g711AToggle        = 0x55
	g711ABase          = 1 << (g711MantissaBits - 1)
	g711ABias          = (1 << bitsPerByte) + g711ABase
	wavSubtypeTail     = "\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
)

type wavLayout struct {
	codec, channels, sampleRate, bits, align uint32
}

func decodeWAV(data []byte, maximumSamples uint64) (DecodedAudio, error) {
	if len(data) < riffHeaderBytes {
		return DecodedAudio{}, io.ErrUnexpectedEOF
	}
	if string(data[:4]) != "RIFF" || string(data[8:riffHeaderBytes]) != "WAVE" {
		return DecodedAudio{}, errors.New("media: audio is not RIFF/WAVE")
	}
	declared := uint64(binary.LittleEndian.Uint32(data[4:8])) + wavChunkHeaderBytes
	if declared > uint64(len(data)) {
		return DecodedAudio{}, io.ErrUnexpectedEOF
	}
	if declared != uint64(len(data)) {
		return DecodedAudio{}, errors.New("media: WAV RIFF size differs from input")
	}
	var layout wavLayout
	var payload []byte
	var formatSeen, dataSeen bool
	for offset := uint64(riffHeaderBytes); offset < declared; {
		if declared-offset < wavChunkHeaderBytes {
			return DecodedAudio{}, io.ErrUnexpectedEOF
		}
		tag := string(data[offset : offset+4])
		size := uint64(binary.LittleEndian.Uint32(data[offset+4 : offset+wavChunkHeaderBytes]))
		body := offset + wavChunkHeaderBytes
		if size > declared-body {
			return DecodedAudio{}, io.ErrUnexpectedEOF
		}
		switch tag {
		case "fmt ":
			if formatSeen {
				return DecodedAudio{}, errors.New("media: duplicate WAV fmt chunk")
			}
			var err error
			layout, err = parseWAVLayout(data[body : body+size])
			if err != nil {
				return DecodedAudio{}, err
			}
			formatSeen = true
		case "data":
			if dataSeen {
				return DecodedAudio{}, errors.New("media: duplicate WAV data chunk")
			}
			payload, dataSeen = data[body:body+size], true
		}
		offset = body + size + size%2
		if offset > declared {
			return DecodedAudio{}, io.ErrUnexpectedEOF
		}
	}
	if !formatSeen || !dataSeen {
		return DecodedAudio{}, errors.New("media: WAV lacks fmt or data")
	}
	if uint64(len(payload))%uint64(layout.align) != 0 {
		return DecodedAudio{}, errors.New("media: incomplete WAV sample frame")
	}
	width := int(layout.bits / bitsPerByte)
	count := len(payload) / width
	if uint64(count) > maximumSamples || count > math.MaxInt/float32Bytes {
		return DecodedAudio{}, errAudioLimit
	}
	audio := DecodedAudio{Format: recipecontract.AudioFormat{
		SampleRate: uint64(layout.sampleRate), Channels: layout.channels, Encoding: "pcm-f32le",
	}, Samples: make([]float32, count)}
	for index := range audio.Samples {
		audio.Samples[index] = decodeWAVScalar(payload[index*width:(index+1)*width], layout)
	}
	return audio, nil
}

func parseWAVLayout(chunk []byte) (wavLayout, error) {
	if len(chunk) < wavFormatMinimumBytes {
		return wavLayout{}, errors.New("media: WAV fmt chunk is too short")
	}
	layout := wavLayout{
		codec: uint32(binary.LittleEndian.Uint16(chunk)), channels: uint32(binary.LittleEndian.Uint16(chunk[2:])),
		sampleRate: binary.LittleEndian.Uint32(chunk[4:]), align: uint32(binary.LittleEndian.Uint16(chunk[12:])),
		bits: uint32(binary.LittleEndian.Uint16(chunk[14:])),
	}
	if layout.codec == wavExtensible {
		if len(chunk) < wavExtensibleBytes || binary.LittleEndian.Uint16(chunk[16:]) < wavExtensionBytes ||
			uint64(binary.LittleEndian.Uint16(chunk[16:])) > uint64(len(chunk)-18) {
			return wavLayout{}, errors.New("media: incomplete WAV extensible format")
		}
		if string(chunk[26:wavExtensibleBytes]) != wavSubtypeTail {
			return wavLayout{}, fmt.Errorf("%w: WAV subtype GUID", errAudioUnsupported)
		}
		if uint32(binary.LittleEndian.Uint16(chunk[18:])) != layout.bits {
			return wavLayout{}, fmt.Errorf("%w: WAV valid bits differ from container bits", errAudioUnsupported)
		}
		mask := binary.LittleEndian.Uint32(chunk[20:])
		if mask != 0 && uint32(bits.OnesCount32(mask)) != layout.channels {
			return wavLayout{}, errors.New("media: WAV channel mask differs from channel count")
		}
		layout.codec = uint32(binary.LittleEndian.Uint16(chunk[24:]))
	}
	supported := layout.codec == wavPCM && (layout.bits == bitsPerByte || layout.bits == pcm16Bits || layout.bits == pcm24Bits || layout.bits == pcm32Bits) ||
		layout.codec == wavIEEEFloat && (layout.bits == float32Bits || layout.bits == float64Bits) ||
		(layout.codec == wavALaw || layout.codec == wavMuLaw) && layout.bits == bitsPerByte
	if !supported {
		return wavLayout{}, fmt.Errorf("%w: WAV %d/%d-bit", errAudioUnsupported, layout.codec, layout.bits)
	}
	if layout.channels == 0 || layout.sampleRate == 0 || layout.align == 0 ||
		layout.align != layout.channels*(layout.bits/bitsPerByte) ||
		uint64(binary.LittleEndian.Uint32(chunk[8:])) != uint64(layout.sampleRate)*uint64(layout.align) {
		return wavLayout{}, errors.New("media: WAV sample geometry is invalid")
	}
	return layout, nil
}

func decodeWAVScalar(data []byte, layout wavLayout) float32 {
	switch layout.codec {
	case wavIEEEFloat:
		if layout.bits == float32Bits {
			return math.Float32frombits(binary.LittleEndian.Uint32(data))
		}
		return float32(math.Float64frombits(binary.LittleEndian.Uint64(data)))
	case wavALaw, wavMuLaw:
		return decodeG711(data[0], layout.codec)
	default:
		if layout.bits == bitsPerByte {
			midpoint := int32(1 << (bitsPerByte - 1))
			return float32(int32(data[0])-midpoint) / float32(midpoint)
		}
		var raw uint32
		for index, value := range data {
			raw |= uint32(value) << (index * bitsPerByte)
		}
		shift := pcm32Bits - layout.bits
		signed := int32(raw<<shift) >> shift
		return float32(float64(signed) / math.Ldexp(1, int(layout.bits)-1))
	}
}

func decodeG711(value byte, codec uint32) float32 {
	if codec == wavMuLaw {
		value = ^value
	} else {
		value ^= g711AToggle
	}
	exponent := (value >> g711MantissaBits) & g711ExponentMask
	mantissa := int(value & g711MantissaMask)
	var magnitude int
	if codec == wavMuLaw {
		magnitude = ((mantissa << (g711MantissaBits - 1)) + g711MuBias) << exponent
		magnitude -= g711MuBias
		if value&g711Sign != 0 {
			magnitude = -magnitude
		}
	} else {
		magnitude = (mantissa << g711MantissaBits) + g711ABase
		if exponent != 0 {
			magnitude = ((mantissa << g711MantissaBits) + g711ABias) << (exponent - 1)
		}
		if value&g711Sign == 0 {
			magnitude = -magnitude
		}
	}
	return float32(magnitude) / pcm16Magnitude
}
