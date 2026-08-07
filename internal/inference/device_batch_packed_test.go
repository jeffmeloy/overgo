package inference

import (
	"reflect"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tokenizer"
)

func TestDecodeCandidatePairsSplitsPackedRows(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{
		CommonSpec: model.CommonSpec{Architecture: "llama", VocabularySize: 5},
	}}}
	runner = attachFixtureProgram(runner)
	sets, err := runner.decodeCandidatePairs(
		[]float32{3, 2.5, 1, 1.5, 4, 3.5, 0, 0.5}, 2, 2,
	)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]LogitCandidate{
		{{ID: 3, Logit: 2.5}, {ID: 1, Logit: 1.5}},
		{{ID: 4, Logit: 3.5}, {ID: 0, Logit: 0.5}},
	}
	if !reflect.DeepEqual(sets, want) {
		t.Fatalf("candidate sets = %v, want %v", sets, want)
	}
}

func TestPlanQwen35DeviceCohorts(t *testing.T) {
	const (
		firstTokens    = 5
		firstPosition  = 7
		secondTokens   = 8
		secondPosition = 10
		pageTokens     = 4
	)
	appends := []deviceBatchAppend{
		{Past: &deviceKVCache{Tokens: firstTokens, Position: firstPosition}, Tokens: []tokenizer.TokenID{1}, PageTokens: pageTokens},
		{Past: &deviceKVCache{Tokens: secondTokens, Position: secondPosition}, Tokens: []tokenizer.TokenID{2}, PageTokens: pageTokens},
		{Past: &deviceKVCache{Tokens: firstTokens, Position: firstPosition}, Tokens: []tokenizer.TokenID{3}, PageTokens: pageTokens},
		{Tokens: []tokenizer.TokenID{4}, PageTokens: pageTokens},
		{Past: &deviceKVCache{Tokens: secondTokens, Position: secondPosition}, Tokens: []tokenizer.TokenID{5}, PageTokens: pageTokens},
		{Past: &deviceKVCache{Tokens: firstTokens, Position: firstPosition}, Tokens: []tokenizer.TokenID{6, 7}, PageTokens: pageTokens},
		{Past: &deviceKVCache{Tokens: firstTokens + 1, Position: firstPosition + 1}, Tokens: []tokenizer.TokenID{8}, PageTokens: pageTokens},
	}
	packed, fallback := planQwen35DeviceCohorts(appends)
	wantPacked := [][]int{{0, 2}, {1, 4}}
	wantFallback := []int{3, 5, 6}
	if !reflect.DeepEqual(packed, wantPacked) || !reflect.DeepEqual(fallback, wantFallback) {
		t.Fatalf("cohorts = %v fallback %v, want %v fallback %v", packed, fallback, wantPacked, wantFallback)
	}
}

func TestPackedDeviceCopyAndSplitTokenCache(t *testing.T) {
	const (
		width         = 4
		heads         = 2
		pastTokens    = 3
		nextTokens    = pastTokens + 1
		sequences     = 2
		f32Bytes      = 4
		firstPointer  = driver.DevicePtr(1024)
		secondPointer = driver.DevicePtr(2048)
		packedPointer = driver.DevicePtr(4096)
	)
	baseShape := tensor.MustShape(width, heads, pastTokens)
	copySpec, err := packedDeviceCopy([]executor.DeviceValue{
		{Pointer: firstPointer, Shape: baseShape},
		{Pointer: secondPointer, Shape: baseShape},
	})
	if err != nil {
		t.Fatal(err)
	}
	wantPackedShape := tensor.MustShape(width, heads, pastTokens, sequences)
	if !copySpec.Shape.Equal(wantPackedShape) || len(copySpec.Segments) != sequences {
		t.Fatalf("packed copy = shape %v segments %d", copySpec.Shape.Slice(), len(copySpec.Segments))
	}
	packed := executor.DeviceValue{
		Pointer: packedPointer, Shape: tensor.MustShape(width, heads, nextTokens, sequences),
	}
	second, err := splitPackedDeviceValue(packed, baseShape, 1, sequences)
	if err != nil {
		t.Fatal(err)
	}
	wantShape := tensor.MustShape(width, heads, nextTokens)
	wantPointer := packedPointer + driver.DevicePtr(width*heads*nextTokens*f32Bytes)
	if !second.Shape.Equal(wantShape) || second.Pointer != wantPointer {
		t.Fatalf("split value = pointer %d shape %v", second.Pointer, second.Shape.Slice())
	}
}

func TestSplitPackedDeviceValueRetainsExplicitSequenceAxis(t *testing.T) {
	const (
		stateWidth    = 2
		heads         = 3
		sequences     = 2
		f32Bytes      = 4
		packedPointer = driver.DevicePtr(8192)
	)
	template := tensor.MustShape(stateWidth, stateWidth, heads, 1)
	packed := executor.DeviceValue{
		Pointer: packedPointer, Shape: tensor.MustShape(stateWidth, stateWidth, heads, sequences),
	}
	second, err := splitPackedDeviceValue(packed, template, 1, sequences)
	if err != nil {
		t.Fatal(err)
	}
	wantPointer := packedPointer + driver.DevicePtr(stateWidth*stateWidth*heads*f32Bytes)
	if !second.Shape.Equal(template) || second.Pointer != wantPointer {
		t.Fatalf("split state = pointer %d shape %v", second.Pointer, second.Shape.Slice())
	}
}

func TestPackedDeviceViewRequiresContiguousSequenceSlabs(t *testing.T) {
	const (
		width        = 4
		heads        = 2
		tokens       = 3
		sequences    = 2
		f32Bytes     = 4
		firstPointer = driver.DevicePtr(16384)
	)
	shape := tensor.MustShape(width, heads, tokens)
	stride := driver.DevicePtr(width * heads * tokens * f32Bytes)
	values := []executor.DeviceValue{
		{Pointer: firstPointer, Shape: shape},
		{Pointer: firstPointer + stride, Shape: shape},
	}
	view, ok := packedDeviceView(values)
	if !ok || view.Pointer != firstPointer ||
		!view.Shape.Equal(tensor.MustShape(width, heads, tokens, sequences)) {
		t.Fatalf("contiguous packed view = %+v, available %t", view, ok)
	}
	values[1].Pointer++
	if _, ok = packedDeviceView(values); ok {
		t.Fatal("non-contiguous device values produced a packed view")
	}
}
