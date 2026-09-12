package devicemath

import (
	"reflect"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/hostmath"
)

func verifyHybridCachedBackward(t *testing.T, worker *device.Worker, x []float32, w hostmath.HybridLayerWeights, d hostmath.HybridLayerDims, state, incoming []float32, want hostmath.HybridDecoderLayerGrads) {
	t.Helper()
	_, cache, err := HybridDecoderLayerForwardDevice(worker, x, w, d, state)
	if err != nil {
		t.Fatal(err)
	}
	got, err := hybridLayerBackwardCachedW(worker, x, hybridHostMatW(w), w, d, state, incoming, &cache)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatal("retained and prepared forward gradients differ")
	}
	if !reflect.DeepEqual(cache, HybridLayerDeviceCache{}) {
		t.Fatal("backward retained its consumed cache")
	}
	if _, err := hybridLayerBackwardCachedW(nil, x, hybridHostMatW(w), w, d, state, incoming, &cache); err == nil {
		t.Fatal("consumed cache reused")
	}
}

func TestHybridBackwardCacheRefusals(t *testing.T) {
	d := hostmath.HybridLayerDims{Tokens: 1, Hidden: 1, Inter: 1}
	one := []float32{1}
	for name, cache := range map[string]*HybridLayerDeviceCache{
		"absent":           nil,
		"empty":            {},
		"wrong-geometry":   {ready: true, dims: hostmath.HybridLayerDims{Tokens: 2}},
		"wrong-mix":        {ready: true, dims: d, IsLinear: true},
		"missing-residual": {ready: true, dims: d},
		"missing-mlp":      {ready: true, dims: d, Xn: one, H: one, Hn: one},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := hybridLayerBackwardCachedW(nil, one, hybridMatW{}, hostmath.HybridLayerWeights{}, d, nil, one, cache); err == nil {
				t.Fatal("invalid cache accepted")
			}
		})
	}
	if _, err := HybridDecoderLayerBackwardDevice(nil, one, hostmath.HybridLayerWeights{}, d, nil, nil); err == nil {
		t.Fatal("malformed incoming gradient accepted")
	}
}
