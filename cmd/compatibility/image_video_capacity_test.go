package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

const imageVideoCapacitySHA256 = "fbd17f0e5caecda155fb820d68715a2547f413a57dbb3a74acef2215ca92d410"

type mediaCapacityCheck struct {
	Name     string      `json:"name"`
	Evidence artifact.ID `json:"evidence"`
	Command  string      `json:"command"`
	Required []string    `json:"required_tests"`
	Scope    string      `json:"scope"`
}

type mediaCapacityBundle struct {
	Version      uint16                 `json:"version"`
	Source       string                 `json:"source_base"`
	Protocol     artifact.ID            `json:"protocol"`
	ProtocolSHA  string                 `json:"protocol_sha256"`
	Environment  artifact.ID            `json:"environment"`
	Prior        map[string]string      `json:"prior_bundles"`
	Checks       []mediaCapacityCheck   `json:"checks"`
	Harnesses    map[string]artifact.ID `json:"test_sources"`
	Loaders      map[string]artifact.ID `json:"loader_observations"`
	Processing   []artifact.ID          `json:"processing_observations"`
	Native       artifact.ID            `json:"native_observations"`
	Host         artifact.ID            `json:"host_observations"`
	NativeOracle string                 `json:"native_oracle_png"`
	Limitations  []string               `json:"limitations"`
	Failed       []artifact.ID          `json:"failed_attempts"`
	Acquisition  artifact.ID            `json:"native_acquisition_protocol"`
	SourceFiles  map[string]struct {
		Evidence artifact.ID `json:"evidence"`
		SHA256   string      `json:"sha256"`
	} `json:"source_files"`
}

type mediaCapacityEnvironment struct {
	Source    string  `json:"source"`
	Available uint64  `json:"host_available_bytes"`
	Total     uint64  `json:"host_total_bytes"`
	CPUs      float64 `json:"cpu_logical_processors"`
	GPU       string  `json:"gpu"`
}

func checkMediaCapacityBinding(bundle mediaCapacityBundle, protocol, environment []byte) (mediaCapacityEnvironment, error) {
	if !gitauthority.ValidObjectID(bundle.Source) {
		return mediaCapacityEnvironment{}, errors.New("capacity acquisition source is invalid")
	}
	if err := checkMediaProtocolIdentity(protocol, bundle.ProtocolSHA); err != nil {
		return mediaCapacityEnvironment{}, err
	}
	var identity struct {
		Source string `json:"source_base"`
	}
	if err := json.Unmarshal(protocol, &identity); err != nil || identity.Source != bundle.Source {
		return mediaCapacityEnvironment{}, errors.New("capacity acquisition source differs")
	}
	var observed mediaCapacityEnvironment
	if err := json.Unmarshal(environment, &observed); err != nil || observed.Source != bundle.Source || observed.Available == 0 || observed.Available > observed.Total || observed.CPUs == 0 || observed.GPU == "" {
		return mediaCapacityEnvironment{}, errors.New("capacity environment is incomplete")
	}
	return observed, nil
}

type mediaCapacityNativeObservation struct {
	Case      string          `json:"case"`
	Request   json.RawMessage `json:"request"`
	PNG       string          `json:"png_sha256"`
	Wall      uint64          `json:"wall_ns"`
	Allocated uint64          `json:"allocated_bytes"`
	Idle      uint64          `json:"idle_device_bytes"`
	Peak      uint64          `json:"library_peak_device_bytes"`
}

type mediaCapacityHostObservation struct {
	Threads   int             `json:"gomaxprocs"`
	CPUs      int             `json:"logical_cpus"`
	Request   json.RawMessage `json:"request"`
	GIF       string          `json:"gif_sha256"`
	Wall      uint64          `json:"wall_ns"`
	Allocated uint64          `json:"allocated_bytes"`
}

// Go's JSON stream may split a long log line across several output events.
// Reassemble output before extracting its structured measurement lines.
func mediaCapacityMeasurements(t testing.TB, raw []byte) []json.RawMessage {
	t.Helper()
	var output strings.Builder
	for line := range strings.SplitSeq(string(raw), "\n") {
		var event struct{ Output string }
		if json.Unmarshal([]byte(line), &event) == nil {
			output.WriteString(event.Output)
		}
	}
	var rows []json.RawMessage
	for line := range strings.SplitSeq(output.String(), "\n") {
		_, value, found := strings.Cut(line, "MEDIA_CAPACITY ")
		if !found {
			continue
		}
		var row any
		if err := json.Unmarshal([]byte(value), &row); err != nil {
			t.Fatal("incomplete resource measurement", err)
		}
		encoded, err := json.Marshal(row)
		if err != nil {
			t.Fatal(err)
		}
		rows = append(rows, encoded)
	}
	return rows
}

func TestImageVideoCapacityEvidenceRejectsAlteration(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(testutil.RepoRoot(t), "docs/image_video_capacity.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"native_observations", "checks", "prior_bundles", "failed_attempts"} {
		var changed map[string]any
		if err := json.Unmarshal(raw, &changed); err != nil {
			t.Fatal(err)
		}
		delete(changed, field)
		candidate, err := json.Marshal(changed)
		if err != nil {
			t.Fatal(err)
		}
		if checkMediaProtocolIdentity(candidate, imageVideoCapacitySHA256) == nil {
			t.Fatal("altered capacity evidence accepted", field)
		}
	}
}

func TestImageVideoCapacityMeasurementsReassembleOutput(t *testing.T) {
	var stream bytes.Buffer
	encoder := json.NewEncoder(&stream)
	for _, part := range []string{"test.go:1: MEDIA_CAP", "ACITY {\"case\":\"changed\",", "\"wall_ns\":1}\n"} {
		if err := encoder.Encode(struct{ Output string }{part}); err != nil {
			t.Fatal(err)
		}
	}
	rows := mediaCapacityMeasurements(t, stream.Bytes())
	if len(rows) != 1 || string(rows[0]) != `{"case":"changed","wall_ns":1}` {
		t.Fatal("split measurement lost or changed", rows)
	}
}

func TestImageVideoCapacityAcceptance(t *testing.T) {
	root := testutil.RepoRoot(t)
	raw, err := os.ReadFile(filepath.Join(root, "docs/image_video_capacity.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := checkMediaProtocolIdentity(raw, imageVideoCapacitySHA256); err != nil {
		t.Fatal(err)
	}
	var bundle mediaCapacityBundle
	if err := json.Unmarshal(raw, &bundle); err != nil || bundle.Version != 1 || !gitauthority.ValidObjectID(bundle.Source) || len(bundle.Checks) == 0 || len(bundle.Limitations) == 0 {
		t.Fatal("incomplete capacity record", err)
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	read := func(id artifact.ID) []byte {
		t.Helper()
		content, found, err := artifact.ReadContent(t.Context(), store, id)
		if err != nil || !found {
			t.Fatal("missing capacity evidence", id, err)
		}
		return content.Data
	}
	protocol := read(bundle.Protocol)
	current, err := os.ReadFile(filepath.Join(root, "docs/image_video_capacity_protocol.json"))
	if err != nil || checkMediaProtocolIdentity(current, bundle.ProtocolSHA) != nil {
		t.Fatal("capacity protocol changed", err)
	}
	environment, err := checkMediaCapacityBinding(bundle, protocol, read(bundle.Environment))
	if err != nil {
		t.Fatal(err)
	}
	for path, digest := range bundle.Prior {
		data, err := os.ReadFile(filepath.Join(root, path))
		if err != nil || checkMediaProtocolIdentity(data, digest) != nil {
			t.Fatal("retained optimization or validation changed", path, err)
		}
	}
	for digest, id := range bundle.Harnesses {
		if fmt.Sprintf("%x", sha256.Sum256(read(id))) != digest {
			t.Fatal("resource harness identity differs", id)
		}
	}
	if len(bundle.Harnesses) == 0 {
		t.Fatal("missing resource harnesses")
	}
	for path, source := range bundle.SourceFiles {
		if bundle.Harnesses[source.SHA256] != source.Evidence {
			t.Fatal("resource source is not a retained harness", path)
		}
		if strings.HasPrefix(path, "tmp/") {
			continue // Overlay sources live in the evidence store, outside the checkout.
		}
		current, err := os.ReadFile(filepath.Join(root, path))
		if err != nil || !bytes.Equal(bytes.ReplaceAll(current, []byte("\r\n"), []byte("\n")), read(source.Evidence)) {
			t.Fatal("resource harness changed since acquisition", path, err)
		}
	}
	var acquisition struct {
		Source string            `json:"source_base"`
		Files  map[string]string `json:"source_sha256"`
	}
	if err := json.Unmarshal(read(bundle.Acquisition), &acquisition); err != nil || acquisition.Source != bundle.Source || len(acquisition.Files) == 0 {
		t.Fatal("missing native acquisition identity", err)
	}
	for path, digest := range acquisition.Files {
		current, err := os.ReadFile(filepath.Join(root, path))
		if err != nil || fmt.Sprintf("%x", sha256.Sum256(bytes.ReplaceAll(current, []byte("\r\n"), []byte("\n")))) != digest {
			t.Fatal("native acquisition input changed", path, err)
		}
	}
	for _, id := range bundle.Failed {
		report, err := testevidence.GoTestJSONReport(string(read(id)))
		if err == nil && testevidence.RequireComplete(report) == nil {
			t.Fatal("failed attempt was replaced by successful evidence", id)
		}
	}
	measurements := map[string][]json.RawMessage{}
	var assertions map[string]map[string][]string
	if err := jsonfile.DecodeStrict(filepath.Join(root, mediaAssertionsPath), &assertions); err != nil {
		t.Fatal(err)
	}
	for _, check := range bundle.Checks {
		if check.Name == "" || check.Command == "" || check.Scope == "" {
			t.Fatal("incomplete resource execution")
		}
		raw := read(check.Evidence)
		requireMediaTestReceipt(t, check.Name, raw, check.Required, assertions)
		measurements[check.Name] = mediaCapacityMeasurements(t, raw)
	}
	matchMeasurements := func(name string, id artifact.ID) {
		t.Helper()
		raw, err := json.Marshal(measurements[name])
		if err != nil {
			t.Fatal(err)
		}
		var stored any
		if err := json.Unmarshal(read(id), &stored); err != nil {
			t.Fatal(err)
		}
		canonical, err := json.Marshal(stored)
		if err != nil || !bytes.Equal(raw, canonical) {
			t.Fatal("resource rows differ from actual test output", name, err)
		}
	}
	matchMeasurements("SenseNova request transitions", bundle.Native)
	matchMeasurements("Un-0 CPU and shape transitions", bundle.Host)
	var native []mediaCapacityNativeObservation
	if err := json.Unmarshal(read(bundle.Native), &native); err != nil || len(native) != len([]string{"native", "changed", "native-repeat", "changed-repeat", "changed-fresh"}) {
		t.Fatal("incomplete native request transitions", err)
	}
	byCase := map[string]mediaCapacityNativeObservation{}
	for _, row := range native {
		if row.Wall == 0 || row.Allocated == 0 || row.Peak < row.Idle || row.PNG == "" {
			t.Fatal("incomplete native observation", row.Case)
		}
		byCase[row.Case] = row
	}
	if byCase["native"].PNG != bundle.NativeOracle || bundle.NativeOracle == "" {
		t.Fatal("native resource case lost its prior oracle")
	}
	for _, pair := range [][2]string{{"native", "native-repeat"}, {"changed", "changed-repeat"}, {"changed", "changed-fresh"}} {
		before, after := byCase[pair[0]], byCase[pair[1]]
		if before.PNG == "" || before.PNG != after.PNG || !bytes.Equal(before.Request, after.Request) {
			t.Fatal("request transition changed output", pair)
		}
	}
	if byCase["changed"].Idle != byCase["changed-repeat"].Idle || bytes.Equal(byCase["native"].Request, byCase["changed"].Request) {
		t.Fatal("shape coverage or stable reusable memory is absent")
	}
	var host []mediaCapacityHostObservation
	if err := json.Unmarshal(read(bundle.Host), &host); err != nil || len(host) == 0 {
		t.Fatal("missing CPU observations", err)
	}
	profiles, outputs := map[int]bool{}, map[string]string{}
	for _, row := range host {
		if row.Wall == 0 || row.Allocated == 0 || row.CPUs != int(environment.CPUs) || row.GIF == "" {
			t.Fatal("invalid CPU observation")
		}
		profiles[row.Threads] = true
		key := string(row.Request)
		if before, found := outputs[key]; found && before != row.GIF {
			t.Fatal("CPU or shape transition changed clip bytes")
		}
		outputs[key] = row.GIF
	}
	if !profiles[1] || !profiles[int(environment.CPUs)] || len(outputs) < 2 {
		t.Fatal("CPU or changed-shape coverage is absent")
	}
	var accepted mediaSharedComponents
	if err := jsonfile.Decode(filepath.Join(root, "docs/image_video_shared_components.json"), &accepted); err != nil {
		t.Fatal(err)
	}
	if len(bundle.Loaders) != len(accepted.Acquisitions) {
		t.Fatal("incomplete complete-loader coverage")
	}
	for _, prior := range accepted.Acquisitions {
		var before, after []mediaLoaderObservation
		if err := json.Unmarshal(read(prior.Observations), &before); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(read(bundle.Loaders[prior.Consumer]), &after); err != nil || checkMediaLoaderObservations(prior.Consumer, after) != nil || len(before) != len(after) {
			t.Fatal("invalid current loader observations", err)
		}
		for index, row := range before {
			current := after[index]
			if current.Values != row.Values || current.Elements != row.Elements || current.Tensors != row.Tensors || current.Source != row.Source {
				t.Fatal("combined implementation changed loaded tensors", prior.Consumer)
			}
		}
	}
	var first mediaProcessingObservation
	for index, id := range bundle.Processing {
		var row mediaProcessingObservation
		if err := json.Unmarshal(read(id), &row); err != nil || row.Wall == 0 || len(row.Samples) == 0 || row.Report == "" {
			t.Fatal("invalid current report/export observations", err)
		}
		if index == 0 {
			first = row
		} else if row.Head != first.Head || row.Sequence != first.Sequence || row.Report != first.Report || !maps.Equal(row.Samples, first.Samples) {
			t.Fatal("report/export repeat changed its store or outputs")
		}
	}
	if len(bundle.Processing) != len([]string{"first", "repeat"}) {
		t.Fatal("missing report/export repeat")
	}
	var merged mediaMergedEvidence
	if err := jsonfile.Decode(filepath.Join(root, "docs/image_video_merged.json"), &merged); err != nil {
		t.Fatal(err)
	}
	if err := checkMediaRuntimeAtRevision(root, "", merged.RuntimePaths, merged.RuntimeSHA256); err != nil {
		t.Fatal(err)
	}
}
