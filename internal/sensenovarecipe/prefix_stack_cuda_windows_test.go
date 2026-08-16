//go:build windows

package sensenovarecipe

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/jsonfile"
	"overgo/internal/routedlm"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

type prefixProbe struct {
	Elements int       `json:"elements"`
	Indices  []int     `json:"indices"`
	Values   []float32 `json:"values"`
}

type prefixLayerOracle struct {
	Hidden prefixProbe `json:"hidden"`
	Key    prefixProbe `json:"key"`
	Value  prefixProbe `json:"value"`
}

type prefixBranchOracle struct {
	IDElements int                          `json:"ids_elements"`
	IDSHA      string                       `json:"ids_sha256_le_i64"`
	ImageTime  int                          `json:"image_time"`
	Layers     map[string]prefixLayerOracle `json:"layers"`
}

type prefixOracle struct {
	Schema   string                        `json:"schema"`
	Branches map[string]prefixBranchOracle `json:"branches"`
}

type generationTextOracle struct {
	TextInputs struct {
		Conditional   []int `json:"conditional_ids"`
		Unconditional []int `json:"unconditional_ids"`
	} `json:"text_inputs"`
}

func TestSenseNovaPrefixStackMatchesOracle(t *testing.T) {
	cudatest.Require(t)
	if os.Getenv("OVERGO_SENSENOVA_BASELINE") != "1" {
		t.Skip("set OVERGO_SENSENOVA_BASELINE=1 for SenseNova prefix evidence")
	}
	var oracle prefixOracle
	if err := jsonfile.Decode(filepath.Join("..", "..", "fixtures", "sensenova", "prefix_oracle.json"), &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.Schema != "overgo.sensenova-prefix-oracle.v1" {
		t.Fatalf("prefix oracle schema=%q", oracle.Schema)
	}
	var generation generationTextOracle
	if err := jsonfile.Decode(senseNovaGenerationGold, &generation); err != nil {
		t.Fatal(err)
	}
	binding := routedlm.SenseNovaBinding()
	cfg, err := routedlm.LoadConfig(senseNovaModelDir, binding)
	if err != nil {
		t.Fatal(err)
	}
	source, err := safetensors.OpenSource(senseNovaModelDir)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	rope, err := routedlm.CompileRopePlan(source, cfg, binding)
	if err != nil {
		t.Fatal(err)
	}
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	cuda, err := executor.NewWithWorker(worker)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	started := time.Now()
	states := make(map[string]*routedlm.PrefixState)
	for name, ids := range map[string][]int{
		"conditional": generation.TextInputs.Conditional, "unconditional": generation.TextInputs.Unconditional,
	} {
		states[name] = runSenseNovaPrefixBranch(t, worker, cuda, source, cfg, binding, rope, name, ids, oracle.Branches[name])
	}
	for name, state := range states {
		if !state.Complete() || len(state.Layers) != cfg.NumHiddenLayers {
			t.Fatalf("%s prefix state is incomplete", name)
		}
	}
	memory, err := worker.MemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("SenseNova neutral prefix stack: branches=2 layers=%d wall=%.3fs peak=%.3fGiB",
		cfg.NumHiddenLayers, time.Since(started).Seconds(), float64(memory.PeakBytes)/(1<<30))
}

func runSenseNovaPrefixBranch(
	t *testing.T,
	worker *device.Worker,
	cuda *executor.Executor,
	source *safetensors.Source,
	cfg routedlm.Config,
	binding routedlm.BranchBinding,
	rope routedlm.RopePlan,
	name string,
	ids []int,
	oracle prefixBranchOracle,
) *routedlm.PrefixState {
	t.Helper()
	if len(ids) != oracle.IDElements || intSHA256(ids) != oracle.IDSHA || oracle.ImageTime != len(ids) {
		t.Fatalf("%s input identity differs", name)
	}
	positions := make([]routedlm.RowPosition, len(ids))
	for index := range positions {
		positions[index] = routedlm.RowPosition{Time: index}
	}
	graph, err := routedlm.BuildDevicePrefixLayer(cfg, rope, positions)
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := executor.Compile(graph.Output, graph.KeyKV, graph.ValueKV)
	if err != nil {
		t.Fatal(err)
	}
	row, err := routedlm.EmbeddingRows(source, cfg, binding, ids)
	if err != nil {
		t.Fatal(err)
	}
	state, err := routedlm.NewPrefixState(
		len(ids), cfg.NumKeyValueHeads, cfg.HeadDim, oracle.ImageTime, cfg.NumHiddenLayers,
	)
	if err != nil {
		t.Fatal(err)
	}
	binder := prefixWeightBinder{worker: worker}
	defer binder.free()
	for layer := 0; layer < cfg.NumHiddenLayers; layer++ {
		weights, err := routedlm.LoadLayerWeights(source, cfg, binding, layer)
		if err != nil {
			t.Fatal(err)
		}
		feeds := make(map[*tensor.Tensor]driver.DevicePtr)
		if err := binder.bind(feeds, graph.Weights, weights); err != nil {
			t.Fatal(err)
		}
		inputs := compiled.NewDeviceInputs()
		for node, pointer := range feeds {
			if err := inputs.Set(node, pointer); err != nil {
				t.Fatal(err)
			}
		}
		result, err := cuda.ExecuteCompiled(
			context.Background(), compiled,
			map[*tensor.Tensor]reference.Value{
				graph.Row: {Shape: graph.Row.Shape, Data: row},
			},
			inputs,
		)
		if err != nil {
			binder.free()
			t.Fatalf("%s layer %d: %v", name, layer, err)
		}
		row = result[graph.Output].Data
		if err := state.SetLayer(layer, result[graph.KeyKV].Data, result[graph.ValueKV].Data); err != nil {
			t.Fatal(err)
		}
		if want, ok := oracle.Layers[fmt.Sprint(layer)]; ok {
			checkPrefixProbe(t, name+" hidden", row, want.Hidden)
			checkPrefixProbe(t, name+" key", result[graph.KeyKV].Data, want.Key)
			checkPrefixProbe(t, name+" value", result[graph.ValueKV].Data, want.Value)
		}
		binder.free()
	}
	return state
}

type prefixWeightBinder struct {
	worker *device.Worker
	ptrs   []driver.DevicePtr
}

func (b *prefixWeightBinder) bind(
	feeds map[*tensor.Tensor]driver.DevicePtr,
	nodes routedlm.DevicePrefillBranch,
	w routedlm.LayerWeights,
) error {
	vector := func(node *tensor.Tensor, values []float32) error {
		return b.upload(feeds, node, driver.Bytes(values))
	}
	matrix := func(node *tensor.Tensor, value routedlm.BF16Matrix) error {
		return b.upload(feeds, node, driver.Bytes(value.Data))
	}
	for _, bind := range []func() error{
		func() error { return vector(nodes.InputNorm, w.InputNorm.Text) },
		func() error { return matrix(nodes.Q, w.QKV.QText) },
		func() error { return matrix(nodes.K, w.QKV.KText) },
		func() error { return matrix(nodes.V, w.QKV.VText) },
		func() error { return matrix(nodes.O, w.QKV.OText) },
		func() error { return vector(nodes.QNorm, w.QKV.QNorm[0]) },
		func() error { return vector(nodes.KNorm, w.QKV.KNorm[0]) },
		func() error { return vector(nodes.PostNorm, w.Output.PostText) },
		func() error { return matrix(nodes.Gate, w.Output.GateText) },
		func() error { return matrix(nodes.Up, w.Output.UpText) },
		func() error { return matrix(nodes.Down, w.Output.DownText) },
	} {
		if err := bind(); err != nil {
			return err
		}
	}
	return nil
}

func (b *prefixWeightBinder) upload(
	feeds map[*tensor.Tensor]driver.DevicePtr,
	node *tensor.Tensor,
	raw []byte,
) error {
	return b.worker.Do(context.Background(), func(state *device.State) error {
		pointer, err := state.Driver.MemAlloc(uint64(len(raw)))
		if err != nil {
			return err
		}
		if err := state.Driver.MemcpyHtoD(pointer, raw); err != nil {
			_ = state.Driver.MemFree(pointer)
			return err
		}
		b.ptrs = append(b.ptrs, pointer)
		feeds[node] = pointer
		return nil
	})
}

func (b *prefixWeightBinder) free() {
	_ = b.worker.Do(context.Background(), func(state *device.State) error {
		for _, pointer := range b.ptrs {
			_ = state.Driver.MemFree(pointer)
		}
		return nil
	})
	b.ptrs = b.ptrs[:0]
}

func checkPrefixProbe(t *testing.T, name string, got []float32, want prefixProbe) {
	t.Helper()
	if len(got) != want.Elements || len(want.Indices) != len(want.Values) || len(want.Indices) == 0 {
		t.Fatalf("%s invalid elements or probes", name)
	}
	worst := 0.0
	dot, gotNorm, wantNorm := 0.0, 0.0, 0.0
	for probe, index := range want.Indices {
		gotValue, wantValue := float64(got[index]), float64(want.Values[probe])
		delta := math.Abs(gotValue - wantValue)
		worst = max(worst, delta)
		dot += gotValue * wantValue
		gotNorm += gotValue * gotValue
		wantNorm += wantValue * wantValue
	}
	cosine := dot / math.Sqrt(gotNorm*wantNorm)
	const directionFloor = 1 - 8.0/256
	if cosine < directionFloor {
		t.Fatalf("%s cosine=%.9f floor=%.9f", name, cosine, directionFloor)
	}
	for index, value := range got {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("%s value[%d]=%g", name, index, value)
		}
	}
	t.Logf("%s probes=%d cosine=%.9f worst=%.3e", name, len(want.Indices), cosine, worst)
}

func intSHA256(values []int) string {
	hash := sha256.New()
	var raw [8]byte
	for _, value := range values {
		binary.LittleEndian.PutUint64(raw[:], uint64(value))
		_, _ = hash.Write(raw[:])
	}
	return hex.EncodeToString(hash.Sum(nil))
}
