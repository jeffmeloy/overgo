package projector

import "llamacpp2go/internal/media"

const (
	wavPCM       = 1
	wavIEEEFloat = 3
)

func DecodeFloat32LE(data []byte) ([]float32, error) {
	return media.DecodeFloat32LE(data)
}

func DecodeWAV(data []byte) ([]float32, int, error) {
	return media.DecodeWAV(data)
}
