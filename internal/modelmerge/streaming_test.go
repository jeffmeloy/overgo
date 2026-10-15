package modelmerge

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	"overgo/internal/recipe"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/testutil"
)

func TestStreamingPassthrough(t *testing.T) {
	// The larger tensor exceeds io.Copy's 32 KiB transfer buffer.
	for _, count := range []int{2, 1 << 16} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			values := make([]float32, count)
			for i := range values {
				values[i] = float32(i)
			}
			model := testutil.ArtifactID(t, artifact.KindModel, "stream passthrough model")
			source := writeStreamingFixture(t, map[string][]float32{"layer.0": values, "layer.1": values})
			plan := sealStreamingPlan(t, composition.OfflineArtifactExactPassthrough, []artifact.ID{model}, []float64{1}, []streamingFixtureTensor{
				{name: "layer.0", shape: []uint64{uint64(count)}}, {name: "layer.1", shape: []uint64{uint64(count)}},
			})
			destination := filepath.Join(t.TempDir(), "output")
			result, err := ExecuteStreaming(t.Context(), plan, []StreamingSource{{Model: model, Directory: source}}, destination)
			if err != nil {
				t.Fatal(err)
			}
			wantResident := safetensors.StreamingBufferBytes(uint64(count * binary.Size(float32(0))))
			if result.TensorCount != len(plan.Operations) || result.PeakResidentBytes != wantResident {
				t.Fatalf("streaming result = %+v, want %d resident payload bytes", result, wantResident)
			}
			for _, name := range []string{"layer.0", "layer.1"} {
				if got := readStreamingTensor(t, destination, name); !slices.Equal(got, values) {
					t.Fatalf("passthrough %s differs", name)
				}
			}
		})
	}
}

func TestStreamingTaskArithmetic(t *testing.T) {
	models := []artifact.ID{
		testutil.ArtifactID(t, artifact.KindModel, "stream arithmetic first"),
		testutil.ArtifactID(t, artifact.KindModel, "stream arithmetic second"),
	}
	first := writeStreamingFixture(t, map[string][]float32{"weight": {2, 4, 6}})
	second := writeStreamingFixture(t, map[string][]float32{"weight": {10, 20, 30}})
	plan := sealStreamingPlan(t, composition.OfflineArtifactTaskArithmetic, models, []float64{0.25, 0.75},
		[]streamingFixtureTensor{{name: "weight", shape: []uint64{3}}})
	destination := filepath.Join(t.TempDir(), "output")
	_, err := ExecuteStreaming(t.Context(), plan, []StreamingSource{
		{Model: models[1], Directory: second},
		{Model: models[0], Directory: first},
	}, destination)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{8, 16, 24}
	if values := readStreamingTensor(t, destination, "weight"); !slices.Equal(values, want) {
		t.Fatalf("task arithmetic = %v, want %v", values, want)
	}
}

func TestStreamingBoundedResidency(t *testing.T) {
	models := []artifact.ID{
		testutil.ArtifactID(t, artifact.KindModel, "stream bounded first"),
		testutil.ArtifactID(t, artifact.KindModel, "stream bounded second"),
		testutil.ArtifactID(t, artifact.KindModel, "stream bounded third"),
	}
	sources := make([]StreamingSource, len(models))
	for index, model := range models {
		sources[index] = StreamingSource{
			Model: model,
			Directory: writeStreamingFixture(t, map[string][]float32{
				"weight": {float32(index + tensor.SingletonExtent), 2},
			}),
		}
	}
	plan := sealStreamingPlan(t, composition.OfflineArtifactTaskArithmetic, models, []float64{0.25, 0.5, 0.25},
		[]streamingFixtureTensor{{name: "weight", shape: []uint64{2}}})
	result, err := ExecuteStreaming(t.Context(), plan, sources, filepath.Join(t.TempDir(), "output"))
	if err != nil {
		t.Fatal(err)
	}
	if result.PeakResidentBytes != plan.PeakResidentBytes || result.PeakResidentBytes != plan.Operations[0].ResidentBytes {
		t.Fatalf("peak residency = %d, plan = %d", result.PeakResidentBytes, plan.PeakResidentBytes)
	}
}

func TestStreamingRefusal(t *testing.T) {
	model := testutil.ArtifactID(t, artifact.KindModel, "stream refusal model")
	source := writeStreamingFixture(t, map[string][]float32{"weight": {1, 2}})
	plan := sealStreamingPlan(t, composition.OfflineArtifactExactPassthrough, []artifact.ID{model}, []float64{1},
		[]streamingFixtureTensor{{name: "weight", shape: []uint64{2}}})
	destination := filepath.Join(t.TempDir(), "wrong-model")
	wrong := testutil.ArtifactID(t, artifact.KindModel, "stream refusal wrong model")
	if _, err := ExecuteStreaming(t.Context(), plan,
		[]StreamingSource{{Model: wrong, Directory: source}}, destination); err == nil {
		t.Fatal("mismatched source identity accepted")
	}
	if _, err := os.Stat(destination); !os.IsNotExist(err) {
		t.Fatalf("refused destination became visible: %v", err)
	}

	tampered := plan
	tampered.Operations = slices.Clone(plan.Operations)
	tampered.Operations[0].OutputBytes += uint64(tensor.SingletonExtent)
	if _, err := ExecuteStreaming(t.Context(), tampered,
		[]StreamingSource{{Model: model, Directory: source}}, filepath.Join(t.TempDir(), "tampered")); err == nil {
		t.Fatal("mismatched plan identity accepted")
	}
}

func TestStreamingRefusesSameGeometrySourceShardByteSubstitution(t *testing.T) {
	model := testutil.ArtifactID(t, artifact.KindModel, "stream shard authority model")
	authoritative := writeStreamingFixture(t, map[string][]float32{"weight": {1, 2}})
	substituted := writeStreamingFixture(t, map[string][]float32{"weight": {3, 4}})
	authoritativeBytes, err := os.ReadFile(filepath.Join(authoritative, streamingSingleShardName))
	if err != nil {
		t.Fatal(err)
	}
	authoritativeID, err := artifact.IdentifyBytes(artifact.KindTensorSet, authoritativeBytes)
	if err != nil {
		t.Fatal(err)
	}
	substitutedBytes, err := os.ReadFile(filepath.Join(substituted, streamingSingleShardName))
	if err != nil {
		t.Fatal(err)
	}
	substitutedID, err := artifact.IdentifyBytes(artifact.KindTensorSet, substitutedBytes)
	if err != nil {
		t.Fatal(err)
	}
	if substitutedID == authoritativeID {
		t.Fatal("same-geometry substitution retained the authoritative byte identity")
	}
	plan := sealStreamingPlan(
		t, composition.OfflineArtifactExactPassthrough, []artifact.ID{model}, []float64{1},
		[]streamingFixtureTensor{{name: "weight", shape: []uint64{2}}},
	)
	destination := filepath.Join(t.TempDir(), "substituted-output")
	if _, err := ExecuteStreaming(t.Context(), plan, []StreamingSource{{
		Model: model, Directory: substituted,
		Shards: []StreamingShard{{Name: streamingSingleShardName, Artifact: authoritativeID}},
	}}, destination); err == nil {
		t.Fatal("same-geometry source shard byte substitution accepted")
	}
	if _, err := os.Stat(destination); !os.IsNotExist(err) {
		t.Fatalf("refused substituted destination became visible: %v", err)
	}
}

type streamingFixtureTensor struct {
	name  string
	shape []uint64
}

func sealStreamingPlan(
	t testing.TB,
	operator composition.OfflineArtifactOperator,
	models []artifact.ID,
	coefficients []float64,
	tensors []streamingFixtureTensor,
) composition.OfflineTensorExecutionPlan {
	t.Helper()
	inputs := make([]composition.OfflineArtifactModel, len(models))
	for index, model := range models {
		inputs[index] = composition.OfflineArtifactModel{
			Model:       model,
			Definition:  testutil.ArtifactID(t, artifact.KindModelDefinition, "stream definition "+model.String()),
			Profile:     testutil.ArtifactID(t, artifact.KindProfile, "stream profile "+model.String()),
			Inventory:   testutil.ArtifactID(t, artifact.KindTensorInventory, "stream inventory "+model.String()),
			Coefficient: coefficients[index],
		}
	}
	operations := make([]composition.OfflineTensorOperation, len(tensors))
	shard := composition.OfflineTensorShard{Index: uint32(tensor.FirstOffset)}
	var peak uint64
	for index, fixture := range tensors {
		bytes, err := safetensors.TensorBytes("F32", fixture.shape)
		if err != nil {
			t.Fatal(err)
		}
		resident := safetensors.StreamingBufferBytes(bytes)
		if operator == composition.OfflineArtifactTaskArithmetic {
			resident = bytes * uint64(tensor.PairedExtent)
		}
		if resident > peak {
			peak = resident
		}
		sourceBytes := make([]uint64, len(models))
		for sourceIndex := range sourceBytes {
			sourceBytes[sourceIndex] = bytes
		}
		position := uint32(index)
		operations[index] = composition.OfflineTensorOperation{
			Index: position, Name: fixture.name, Storage: "f32", SourceBytes: sourceBytes,
			OutputBytes: bytes, ResidentBytes: resident, Placement: recipe.PlacementHost,
			Lifetime: composition.OfflineTensorLifetime{First: position, Last: position},
			Shard:    uint32(tensor.FirstOffset),
		}
		shard.Bytes += bytes
		shard.Tensors = append(shard.Tensors, fixture.name)
	}
	plan := composition.OfflineTensorExecutionPlan{
		Version:        artifact.InitialDocumentVersion,
		ArtifactPlan:   testutil.ArtifactID(t, artifact.KindRecipe, "stream artifact plan"),
		ResourcePolicy: testutil.ArtifactID(t, artifact.KindProfile, "stream resource policy"),
		Operator:       operator, Inputs: inputs, Operations: operations,
		Shards: []composition.OfflineTensorShard{shard}, PeakResidentBytes: peak,
	}
	data, err := json.Marshal(plan)
	if err != nil {
		t.Fatal(err)
	}
	plan.ID, err = artifact.IdentifyBytes(artifact.KindProfile, data)
	if err != nil {
		t.Fatal(err)
	}
	if err := plan.ValidateIdentity(); err != nil {
		t.Fatal(err)
	}
	return plan
}

func writeStreamingFixture(t testing.TB, values map[string][]float32) string {
	t.Helper()
	directory := t.TempDir()
	shapes := make(map[string][]int, len(values))
	for name, tensorValues := range values {
		shapes[name] = []int{len(tensorValues)}
	}
	if err := safetensors.Save(filepath.Join(directory, streamingSingleShardName), values, shapes, nil); err != nil {
		t.Fatal(err)
	}
	return directory
}

func readStreamingTensor(t *testing.T, directory, name string) []float32 {
	t.Helper()
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	tensorView, found := source.Tensors[name]
	if !found {
		t.Fatalf("tensor %q is absent", name)
	}
	data := make([]byte, tensorView.Size())
	if _, err := tensorView.Reader().Read(data); err != nil {
		t.Fatal(err)
	}
	values := make([]float32, len(data)/binary.Size(float32(0)))
	for index := range values {
		values[index] = math.Float32frombits(binary.LittleEndian.Uint32(data[index*binary.Size(float32(0)):]))
	}
	return values
}

func BenchmarkStreamingPassthrough(b *testing.B) {
	// Four MiB exposes the displaced tensor allocation without a model dependency.
	values := make([]float32, 1<<20)
	model := testutil.ArtifactID(b, artifact.KindModel, "passthrough benchmark")
	source := writeStreamingFixture(b, map[string][]float32{"weight": values})
	plan := sealStreamingPlan(b, composition.OfflineArtifactExactPassthrough, []artifact.ID{model}, []float64{1}, []streamingFixtureTensor{{name: "weight", shape: []uint64{uint64(len(values))}}})
	destination := filepath.Join(b.TempDir(), "output")
	b.SetBytes(int64(len(values) * binary.Size(float32(0))))
	b.ReportAllocs()
	for b.Loop() {
		if _, err := ExecuteStreaming(b.Context(), plan, []StreamingSource{{Model: model, Directory: source}}, destination); err != nil {
			b.Fatal(err)
		}
		if err := os.RemoveAll(destination); err != nil {
			b.Fatal(err)
		}
	}
}
