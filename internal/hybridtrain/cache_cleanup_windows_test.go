package hybridtrain

import (
	"reflect"
	"testing"

	"overgo/internal/devicemath"
	"overgo/internal/hostmath"
)

func TestResidentTrainingCacheCleanup(t *testing.T) {
	model, err := BuildModel(smallHybrid(), testSeed)
	if err != nil {
		t.Fatal(err)
	}
	training := &residentTraining{
		model:     model,
		weights:   make([]devicemath.HybridLayerResidentMatrices, len(model.Weights)),
		gradients: make([]devicemath.HybridLayerResidentMatrices, len(model.Weights)),
		inputs:    make([][]float32, len(model.Weights)),
		caches:    make([]devicemath.HybridLayerDeviceCache, len(model.Weights)),
		grads:     make([]hostmath.HybridDecoderLayerGrads, len(model.Weights)),
	}
	for index := range model.Weights {
		training.inputs[index] = model.X
		training.caches[index].Xn = model.X
		training.grads[index].DX = model.X
	}
	// Absent matrix bindings refuse before accessing the nil worker. Already
	// retained references must still be released when backward cannot finish.
	if err := training.Backward(model.Target); err == nil {
		t.Fatal("missing resident bindings accepted")
	}
	for index := range model.Weights {
		if training.inputs[index] != nil || !reflect.DeepEqual(training.caches[index], devicemath.HybridLayerDeviceCache{}) || !reflect.DeepEqual(training.grads[index], hostmath.HybridDecoderLayerGrads{}) {
			t.Fatal("failed backward retained activation or gradient references")
		}
	}
}
