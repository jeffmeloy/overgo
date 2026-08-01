package projector

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

const (
	wavPCM       = 1
	wavIEEEFloat = 3
)

func DecodeFloat32LE(data []byte) ([]float32, error) {
	if len(data) == 0 {
		return nil, errors.New("projector: audio is empty")
	}
	if len(data)%4 != 0 {
		return nil, fmt.Errorf("projector: float32 audio byte count %d is not divisible by 4", len(data))
	}
	samples := make([]float32, len(data)/4)
	for index := range samples {
		samples[index] = math.Float32frombits(binary.LittleEndian.Uint32(data[index*4:]))
	}
	return samples, nil
}

func DecodeWAV(data []byte) ([]float32, int, error) {
	if len(data) < 12 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, 0, errors.New("projector: audio is not RIFF/WAVE")
	}
	var format, channels, sampleRate, bitsPerSample, blockAlign int
	var payload []byte
	for offset := 12; offset+8 <= len(data); {
		size := uint64(binary.LittleEndian.Uint32(data[offset+4 : offset+8]))
		body := uint64(offset + 8)
		end := body + size
		if end > uint64(len(data)) {
			return nil, 0, fmt.Errorf("projector: truncated WAV %q chunk", string(data[offset:offset+4]))
		}
		switch string(data[offset : offset+4]) {
		case "fmt ":
			if size < 16 {
				return nil, 0, errors.New("projector: WAV fmt chunk is too short")
			}
			chunk := data[body:end]
			format = int(binary.LittleEndian.Uint16(chunk[0:2]))
			channels = int(binary.LittleEndian.Uint16(chunk[2:4]))
			sampleRate = int(binary.LittleEndian.Uint32(chunk[4:8]))
			blockAlign = int(binary.LittleEndian.Uint16(chunk[12:14]))
			bitsPerSample = int(binary.LittleEndian.Uint16(chunk[14:16]))
		case "data":
			payload = data[body:end]
		}
		next := end + size%2
		if next > uint64(len(data)) || next > uint64(^uint(0)>>1) {
			return nil, 0, errors.New("projector: WAV chunk offset exceeds native limits")
		}
		offset = int(next)
	}
	if format == 0 || sampleRate <= 0 || payload == nil {
		return nil, 0, errors.New("projector: WAV lacks fmt or data")
	}
	if channels != 1 {
		return nil, 0, fmt.Errorf("projector: WAV has %d channels; want mono", channels)
	}
	bytesPerSample := bitsPerSample / 8
	if bitsPerSample%8 != 0 || bytesPerSample <= 0 || blockAlign != bytesPerSample || len(payload)%blockAlign != 0 {
		return nil, 0, errors.New("projector: WAV sample layout is invalid")
	}
	samples := make([]float32, len(payload)/blockAlign)
	switch {
	case format == wavPCM && bitsPerSample == 16:
		for index := range samples {
			samples[index] = float32(int16(binary.LittleEndian.Uint16(payload[index*2:]))) / 32768
		}
	case format == wavIEEEFloat && bitsPerSample == 32:
		for index := range samples {
			samples[index] = math.Float32frombits(binary.LittleEndian.Uint32(payload[index*4:]))
		}
	default:
		return nil, 0, fmt.Errorf("projector: WAV format %d/%d-bit is unsupported", format, bitsPerSample)
	}
	return samples, sampleRate, nil
}
