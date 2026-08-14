package model

import (
	"context"
	"errors"
	"sync"

	"overgo/internal/gguf"
	"overgo/internal/tensor/reference"
)

// HostTensorStore: lazy persistent F32 host weights.
type HostTensorStore struct {
	mu     sync.Mutex
	values map[string]reference.Value
	layers map[string]HostLayer
}

func NewHostTensorStore() *HostTensorStore {
	return &HostTensorStore{
		values: make(map[string]reference.Value),
		layers: make(map[string]HostLayer),
	}
}

func (s *HostTensorStore) Load(
	ctx context.Context,
	file *gguf.File,
	info gguf.TensorInfo,
) (reference.Value, error) {
	if s == nil {
		return reference.Value{}, errors.New("host tensor store is nil")
	}
	if err := ctx.Err(); err != nil {
		return reference.Value{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if value, ok := s.values[info.Name]; ok {
		return value, nil
	}
	value, err := LoadHostTensor(ctx, file, info)
	if err != nil {
		return reference.Value{}, err
	}
	s.values[info.Name] = value
	return value, nil
}

func (s *HostTensorStore) LoadLayer(
	ctx context.Context,
	file *gguf.File,
	key string,
	info LayerWeights,
) (HostLayer, error) {
	if s == nil || key == "" {
		return HostLayer{}, errors.New("host layer store key is invalid")
	}
	s.mu.Lock()
	if layer, ok := s.layers[key]; ok {
		s.mu.Unlock()
		return layer, nil
	}
	s.mu.Unlock()
	var result HostLayer
	if err := loadHostLayerGraphFieldsWith(ctx, file, &info, &result, s.Load); err != nil {
		return HostLayer{}, err
	}
	s.mu.Lock()
	s.layers[key] = result
	s.mu.Unlock()
	return result, nil
}

func (s *HostTensorStore) Release() {
	if s == nil {
		return
	}
	s.mu.Lock()
	clear(s.values)
	clear(s.layers)
	s.mu.Unlock()
}
