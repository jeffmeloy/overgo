package audioparity

import (
	"cmp"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/dataset"
	"overgo/internal/jsonfile"
	"overgo/internal/media"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
	"overgo/internal/speechactivity"
	"overgo/internal/testutil"
)

type vadNumericalCase struct {
	Input          string                 `json:"input"`
	Frames         int                    `json:"frames"`
	Width          int                    `json:"feature_width"`
	SelectedFrames []int                  `json:"selected_frames"`
	Features       [][]float32            `json:"features"`
	Probabilities  []float32              `json:"probabilities"`
	Traces         map[string][][]float32 `json:"traces"`
	CacheChannels  []int                  `json:"cache_channels"`
	Caches         [][][]float32          `json:"caches"`
}

func TestVADWaveformStreamParity(t *testing.T) {
	reference, store, audio := loadVADReference(t)
	var config audiodsp.FrontendConfig
	readVADJSON(t, "recipes/vad_frontend.json", &config)
	offline, err := audiodsp.NewFrontend(config, vadReferenceBytes)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := audiodsp.NewStreamFrontend(config, vadReferenceBytes)
	if err != nil {
		t.Fatal(err)
	}
	executed := 0
	for _, model := range reference.Models {
		network, _ := loadVADNetwork(t, store, model)
		if !network.Causal() {
			continue
		}
		for _, fixture := range model.Cases {
			t.Run(fixture.Input, func(t *testing.T) {
				wave := audio.Samples
				if fixture.Input == "zero-control" {
					wave = make([]float32, len(wave))
				}
				var fw audiodsp.Workspace
				features, frames, err := offline.Process(t.Context(), [][]float32{wave}, config.SampleRate, &fw, audiodsp.ProcessOptions{})
				if err != nil {
					t.Fatal(err)
				}
				var nw speechactivity.Workspace
				want, wantCache, err := network.Evaluate(t.Context(), features, frames, nil, &nw, nil)
				if err != nil {
					t.Fatal(err)
				}
				// One sample smaller than a window and one sample larger than a
				// multi-hop chunk exercise both incomplete input and misalignment.
				for _, chunk := range []int{config.FrameSpan - 1, int(config.Geometry.HopSamples)*len(model.Cases) + 1} {
					var sw audiodsp.StreamWorkspace
					var execution speechactivity.Workspace
					var state audiodsp.StreamState
					var caches [][]float32
					var got []float32
					chunks := 0
					for start := 0; start < len(wave); start += chunk {
						end := min(start+chunk, len(wave))
						values, count, next, err := stream.Process(t.Context(), wave[start:end], config.SampleRate, state, end == len(wave), &sw)
						if err != nil {
							t.Fatal(err)
						}
						state = next
						if count > 0 {
							probabilities, next, err := network.Evaluate(t.Context(), values, count, caches, &execution, nil)
							if err != nil {
								t.Fatal(err)
							}
							got, caches = append(got, probabilities...), next
						}
						chunks++
						// Serialize and replace both execution workspaces at each chunk:
						// no hidden process-local state can supply missing context.
						encoded, err := json.Marshal(struct {
							Frontend audiodsp.StreamState
							Caches   [][]float32
						}{state, caches})
						if err != nil {
							t.Fatal(err)
						}
						var restart struct {
							Frontend audiodsp.StreamState
							Caches   [][]float32
						}
						if err := json.Unmarshal(encoded, &restart); err != nil {
							t.Fatal(err)
						}
						state, caches = restart.Frontend, restart.Caches
						sw, execution = audiodsp.StreamWorkspace{}, speechactivity.Workspace{}
					}
					if !slices.Equal(got, want) || !reflect.DeepEqual(caches, wantCache) || !state.Final || state.Frames != uint64(frames) {
						t.Fatalf("waveform chunk %d differs from whole causal execution", chunk)
					}
					assertASRBoundary(t, "vad/live-probabilities", got, fixture.Probabilities, config.FFTLength)
					t.Logf("CPU frames=%d chunks=%d serialized restarts=%d; bit-exact Go whole/live output and cache; source probability parity checked", frames, chunks, chunks)
				}
			})
			executed++
		}
	}
	if executed != 2 {
		t.Fatalf("streaming model cases=%d, want speech and silence", executed)
	}
}

type vadNumericalModel struct {
	Checkpoint  string             `json:"checkpoint"`
	SHA256      string             `json:"sha256"`
	CMVNSHA256  string             `json:"cmvn_sha256"`
	Declaration string             `json:"declaration"`
	Cases       []vadNumericalCase `json:"cases"`
}

type vadNumericalReference struct {
	Schema         string              `json:"schema"`
	Source         string              `json:"source_commit"`
	FrontendSource string              `json:"frontend_source_commit"`
	Corpus         ctcTrainingRecord   `json:"corpus"`
	Models         []vadNumericalModel `json:"models"`
}

// This is an explicit integration-test numeric ceiling, not a runtime default.
const vadReferenceBytes = 32 << 20

func readVADJSON(t *testing.T, path string, value any) {
	t.Helper()
	if err := jsonfile.Decode(path, value); err != nil {
		t.Fatal(err)
	}
}

func loadVADReference(t *testing.T) (vadNumericalReference, *overgodb.Store, media.DecodedAudio) {
	t.Helper()
	var reference vadNumericalReference
	readVADJSON(t, "testdata/vad_reference.json", &reference)
	if reference.Schema != "overgo/vad-native-reference/v1" || reference.Source != "c30ec49e8cc69642b0ee65362eba11b9d11c6e54" ||
		reference.FrontendSource != "f68c6b43f739697d7ab02ff6debacee130e1d541" || len(reference.Models) != 2 {
		t.Fatal("VAD oracle identities or denominator differ")
	}
	storeRoot := cmp.Or(os.Getenv("OVERGO_AUDIO_REFERENCE_STORE"), filepath.Join(testutil.RepoRoot(t), "overgodb-store"))
	store, err := overgodb.OpenReadOnly(storeRoot)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	var existing ctcTrainingRecord
	readVADJSON(t, "testdata/ctc_training_record.json", &existing)
	record := reference.Corpus
	if record.ShardSHA256 != existing.ShardSHA256 || record.AudioSHA256 != existing.AudioSHA256 || record.RowID != existing.RowID || record.Row != existing.Row || record.Split != existing.Split || record.Shard != existing.Shard {
		t.Fatal("VAD source is not the pinned shared corpus row")
	}
	corpus := filepath.Join(filepath.Dir(storeRoot), "datasets", "librispeech_asr-clean-xet", "clean", record.Split, record.Shard)
	id, err := artifact.ParseID("file:sha256:" + record.ShardSHA256)
	if err != nil {
		t.Fatal(err)
	}
	verifyASRFile(t, corpus, id, record.ShardBytes)
	// The existing Arrow/Parquet owner bounds source materialization separately.
	rows, err := dataset.OpenParquetRows(t.Context(), corpus, []string{"audio.bytes", "id"}, 512<<20)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	row, err := rows.Read(t.Context(), record.Row)
	if err != nil || row["id"] == nil || *row["id"] != record.RowID || row["audio.bytes"] == nil {
		t.Fatalf("VAD corpus row: %v", err)
	}
	encoded := []byte(*row["audio.bytes"])
	id, err = artifact.IdentifyBytes(artifact.KindFile, encoded)
	if err != nil || id.DigestHex() != record.AudioSHA256 {
		t.Fatal("VAD encoded audio differs")
	}
	audio, _, err := media.DecodeAudio(t.Context(), encoded, record.Samples)
	if err != nil {
		t.Fatal(err)
	}
	if uint64(len(audio.Samples)) != record.Samples || audio.Format.SampleRate != record.SampleRate || audio.Format.Channels != 1 {
		t.Fatal("VAD decoded geometry differs")
	}
	return reference, store, audio
}

func loadVADNetwork(t *testing.T, store *overgodb.Store, model vadNumericalModel) (*speechactivity.Network, speechactivity.Declaration) {
	t.Helper()
	_, _, checkpoint, declaration := loadVADArtifacts(t, store, model)
	network, err := speechactivity.LoadNetwork(t.Context(), checkpoint, declaration, vadReferenceBytes)
	if err != nil {
		t.Fatal(err)
	}
	return network, declaration
}

func loadVADArtifacts(t *testing.T, store *overgodb.Store, model vadNumericalModel) (modelartifact.Inventory, artifact.Content, string, speechactivity.Declaration) {
	t.Helper()
	var registrations []struct {
		Model      artifact.ID              `json:"model"`
		Tensors    artifact.ID              `json:"tensor_inventory"`
		Components []modelartifact.FileSpec `json:"components"`
		License    struct {
			Artifact artifact.ID `json:"artifact"`
			Path     string      `json:"path"`
		} `json:"license"`
	}
	readVADJSON(t, "testdata/registered_models.json", &registrations)
	for _, registration := range registrations {
		for _, component := range registration.Components {
			if component.Role != artifact.ComponentWeights || filepath.ToSlash(component.Path) != model.Checkpoint+"/model.pth.tar" {
				continue
			}
			root, err := artifact.AvailablePath(t.Context(), store, registration.Model, artifact.LocationDirectory)
			if err != nil {
				t.Fatal(err)
			}
			license, found, err := artifact.ReadContent(t.Context(), store, registration.License.Artifact)
			if err != nil || !found || len(license.Data) == 0 {
				t.Fatalf("VAD license absent: %v", err)
			}
			verifyASRFile(t, filepath.Join(root, registration.License.Path), registration.License.Artifact, license.Descriptor.Size)
			components := slices.Clone(registration.Components)
			for i := range components {
				components[i].Path = filepath.Join(root, components[i].Path)
			}
			inventory, err := modelartifact.FromFiles(root, components)
			if err != nil || inventory.Manifest.ID != registration.Model || inventory.TensorInventory.ID != registration.Tensors {
				t.Fatalf("VAD physical identity differs: %v", err)
			}
			checkpoint := filepath.Join(root, component.Path)
			id, err := artifact.ParseID("tensor-set:sha256:" + model.SHA256)
			if err != nil {
				t.Fatal(err)
			}
			info, err := os.Stat(checkpoint)
			if err != nil {
				t.Fatal(err)
			}
			verifyASRFile(t, checkpoint, id, uint64(info.Size()))
			var declaration speechactivity.Declaration
			readVADJSON(t, model.Declaration, &declaration)
			return inventory, license, checkpoint, declaration
		}
	}
	t.Fatal("VAD model registration absent")
	return modelartifact.Inventory{}, artifact.Content{}, "", speechactivity.Declaration{}
}

func TestVADNumericalParity(t *testing.T) {
	reference, store, audio := loadVADReference(t)
	var config audiodsp.FrontendConfig
	readVADJSON(t, "recipes/vad_frontend.json", &config)
	frontend, err := audiodsp.NewFrontend(config, vadReferenceBytes)
	if err != nil {
		t.Fatal(err)
	}
	for _, model := range reference.Models {
		t.Run(model.Checkpoint, func(t *testing.T) {
			network, declaration := loadVADNetwork(t, store, model)
			if len(model.Cases) != 2 {
				t.Fatal("VAD case denominator differs")
			}
			for _, fixture := range model.Cases {
				t.Run(fixture.Input, func(t *testing.T) {
					wave := audio.Samples
					if fixture.Input == "zero-control" {
						wave = make([]float32, len(wave))
					} else if fixture.Input != "speech" {
						t.Fatal("unknown VAD case")
					}
					var fw audiodsp.Workspace
					features, frames, err := frontend.Process(t.Context(), [][]float32{wave}, int(audio.Format.SampleRate), &fw, audiodsp.ProcessOptions{})
					if err != nil {
						t.Fatal(err)
					}
					if frames != fixture.Frames || network.InputWidth() != fixture.Width || len(fixture.Probabilities) != frames || len(fixture.SelectedFrames) != 3 || len(fixture.Features) != 3 {
						t.Fatal("VAD numerical denominator differs")
					}
					for index, frame := range fixture.SelectedFrames {
						assertASRBoundary(t, "vad/features", features[frame*fixture.Width:(frame+1)*fixture.Width], fixture.Features[index], config.FFTLength)
					}
					var nw speechactivity.Workspace
					seen := map[string]bool{}
					probabilities, caches, err := network.Evaluate(t.Context(), features, frames, nil, &nw, func(trace speechactivity.Trace) error {
						key := ""
						switch trace.Stage {
						case "expand":
							if trace.Index == 0 {
								key = "expanded"
							}
						case "project":
							if trace.Index == 0 {
								key = "projected"
							}
						case "memory":
							key = fmt.Sprintf("memory_%d", trace.Index)
						case "dense":
							key = "dense"
						case "logits":
							key = "logits"
						}
						if key == "" {
							return nil
						}
						golden, found := fixture.Traces[key]
						if !found || seen[key] || len(golden) != len(fixture.SelectedFrames) {
							return fmt.Errorf("VAD trace %s missing or repeated", key)
						}
						seen[key] = true
						for index, frame := range fixture.SelectedFrames {
							assertASRBoundary(t, "vad/"+key, trace.Values[frame*trace.Width:(frame+1)*trace.Width], golden[index], max(config.FFTLength, trace.Width))
						}
						return nil
					})
					if err != nil {
						t.Fatal(err)
					}
					if len(seen) != len(fixture.Traces) || len(caches) != len(declaration.Blocks) || len(fixture.Caches) != len(caches) {
						t.Fatal("VAD execution or cache coverage incomplete")
					}
					assertASRBoundary(t, "vad/probabilities", probabilities, fixture.Probabilities, config.FFTLength)
					for layer, cache := range caches {
						if len(fixture.Caches[layer]) == 0 || len(cache)%len(fixture.Caches[layer]) != 0 {
							t.Fatal("VAD cache frame geometry differs")
						}
						width := len(cache) / len(fixture.Caches[layer])
						var got, want []float32
						for frame, values := range fixture.Caches[layer] {
							if len(values) != len(fixture.CacheChannels) {
								t.Fatal("VAD cache channel denominator differs")
							}
							for _, channel := range fixture.CacheChannels {
								if channel < 0 || channel >= width {
									t.Fatal("VAD cache channel outside state")
								}
								got = append(got, cache[frame*width+channel])
							}
							want = append(want, values...)
						}
						// Compare the layer state as one boundary, like probabilities and
						// other tensor traces. A three-channel row near cancellation is
						// not an independent scale for accumulated upstream roundoff.
						assertASRBoundary(t, fmt.Sprintf("vad/cache/%d", layer), got, want, config.FFTLength)
					}
					t.Logf("CPU VAD frames=%d; all probabilities, %d intermediate boundaries and %d memory blocks compared; no annotated quality or GPU claim", frames, len(seen), len(caches))
				})
			}
		})
	}
}
