package main

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

const imageVideoRoutedImagesSHA256 = "3703f6ab9ad58b847dab9d0adf0e348e5aeb37f9aea1940b29e5e71ca3d498f9"

type mediaRoutedCase struct {
	Case        string      `json:"case"`
	Mode        string      `json:"mode"`
	Run         artifact.ID `json:"run"`
	Output      artifact.ID `json:"output"`
	Acquisition artifact.ID `json:"acquisition"`
	Review      artifact.ID `json:"review"`
	Test        artifact.ID `json:"test_evidence"`
}
type mediaRoutedBundle struct {
	Version         uint16                 `json:"version"`
	Source          string                 `json:"source_base"`
	Protocol        artifact.ID            `json:"protocol"`
	Cases           []mediaRoutedCase      `json:"cases"`
	Numerical       []mediaSharedCheck     `json:"numerical"`
	Instrumentation map[string]artifact.ID `json:"instrumentation"`
	Failed          []artifact.ID          `json:"failed_attempts"`
	Previous        string                 `json:"previous_source"`
	StablePaths     []string               `json:"stable_paths"`
	StableSHA       string                 `json:"stable_sha256"`
	ConverterBefore string                 `json:"converter_before_sha256"`
	ConverterAfter  string                 `json:"converter_after_sha256"`
	ReuseScope      string                 `json:"reuse_scope"`
	Scope           string                 `json:"scope"`
}

func checkMediaCPUImage(row mediaLatentObservation, run runrecord.Run, source string) error {
	if row.Source != source || run.CodeCommit != source || row.Run != run.ID || row.Recipe != run.Recipe || row.Environment != run.Environment || run.Outcome != runrecord.OutcomeSucceeded || !slices.Equal(run.Inputs, []artifact.ID{row.Input}) || !slices.Equal(run.Outputs, []artifact.ID{row.Output}) {
		return errors.New("CPU image run identity differs")
	}
	if !row.Finite || math.IsNaN(row.Minimum) || math.IsInf(row.Minimum, 0) || math.IsNaN(row.Maximum) || math.IsInf(row.Maximum, 0) || row.Minimum > row.Maximum || row.Wall == 0 || run.MeasuredNS != row.Wall || row.Peak == 0 || row.Allocated == 0 || row.Width <= 0 || row.Height <= 0 || len(run.Phases) == 0 {
		return errors.New("CPU image lacks finite output and measured resources")
	}
	return nil
}

func TestImageVideoRoutedImageAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration + ": retained routed image bundle")
	}
	root := testutil.RepoRoot(t)
	document, err := plan.Load(filepath.Join(root, plan.Path))
	if err != nil {
		t.Fatal(err)
	}
	if document.Lane != "image_video_gen" || os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: explicit media data root required")
	}
	path := filepath.Join(root, "docs/image_video_routed_images.json")
	var bundle mediaRoutedBundle
	if err := jsonfile.DecodeStrict(path, &bundle); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkMediaProtocolIdentity(raw, imageVideoRoutedImagesSHA256); err != nil {
		t.Fatal(err)
	}
	if bundle.Version != artifact.InitialDocumentVersion || bundle.Source == "" || bundle.Previous == "" || bundle.Scope == "" || bundle.ReuseScope == "" || len(bundle.Numerical) != 3 || len(bundle.Failed) != 1 {
		t.Fatal("incomplete routed image bundle")
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	read := func(id artifact.ID) []byte {
		t.Helper()
		content, found, err := artifact.ReadContent(t.Context(), store, id)
		if err != nil || !found {
			t.Fatal("missing routed image evidence", id, err)
		}
		return content.Data
	}
	checkTest := func(id artifact.ID) {
		t.Helper()
		report, err := testevidence.GoTestJSONReport(string(read(id)))
		if err != nil {
			t.Fatal(err)
		}
		if err := testevidence.RequireComplete(report); err != nil {
			t.Fatal(err)
		}
		if report.PassedTests == 0 || report.PassedPackages == 0 {
			t.Fatal("empty routed check")
		}
	}
	for _, revision := range []string{bundle.Previous, bundle.Source, "HEAD"} {
		if err := checkMediaRuntimeAtRevision(root, revision, bundle.StablePaths, bundle.StableSHA); err != nil {
			t.Fatal(err)
		}
	}
	for revision, digest := range map[string]string{bundle.Previous: bundle.ConverterBefore, bundle.Source: bundle.ConverterAfter, "HEAD": bundle.ConverterAfter} {
		command := exec.Command("git", "show", revision+":internal/safetensors/convert.go")
		command.Dir = root
		data, err := command.Output()
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(data)) != digest {
			t.Fatal("unreviewed shared converter change")
		}
	}
	primary := read(bundle.Protocol)
	var protocol struct {
		Source  string `json:"source_base"`
		Quality string `json:"quality_protocol_sha256"`
	}
	if err := json.Unmarshal(primary, &protocol); err != nil {
		t.Fatal(err)
	}
	if protocol.Source != bundle.Source || protocol.Quality != imageVideoProtocolSHA256 {
		t.Fatal("protocol source or criteria differs")
	}
	for _, pair := range [][2]string{{"media-sensenova-acquisition-instrumentation.json", "media_sensenova_acquisition_test.go"}, {"media-un0-acquisition-instrumentation-corrected.json", "media_un0_acquisition_test.go"}} {
		var instrumentation struct {
			Source   string `json:"source_base"`
			Protocol string `json:"protocol_sha256"`
			Harness  string `json:"harness_sha256"`
		}
		if err := json.Unmarshal(read(bundle.Instrumentation[pair[0]]), &instrumentation); err != nil {
			t.Fatal(err)
		}
		if instrumentation.Source != bundle.Source || instrumentation.Protocol != fmt.Sprintf("%x", sha256.Sum256(primary)) || instrumentation.Harness != fmt.Sprintf("%x", sha256.Sum256(read(bundle.Instrumentation[pair[1]]))) {
			t.Fatal("acquisition instrumentation differs")
		}
	}
	for _, id := range bundle.Failed {
		report, err := testevidence.GoTestJSONReport(string(read(id)))
		if err != nil || len(report.Failed) == 0 {
			t.Fatal("lost failed prerequisite attempt", err)
		}
	}
	for _, check := range bundle.Numerical {
		if check.Name == "" || check.Command == "" {
			t.Fatal("missing numerical scope")
		}
		checkTest(check.Evidence)
	}
	quality, err := os.ReadFile(filepath.Join(root, "docs/image_video_protocol.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := checkImageVideoProtocolIdentity(quality); err != nil {
		t.Fatal(err)
	}
	type frozenCase struct {
		ID           string                  `json:"id"`
		Model        artifact.ID             `json:"model"`
		Recipe       artifact.ID             `json:"recipe"`
		Inputs       []artifact.ID           `json:"inputs"`
		Outputs      []artifact.ID           `json:"outputs"`
		Run          artifact.ID             `json:"source_run"`
		Observations []imageVideoObservation `json:"observations"`
	}
	var frozen struct {
		Cases []frozenCase `json:"cases"`
	}
	if err := json.Unmarshal(quality, &frozen); err != nil {
		t.Fatal(err)
	}
	var expected []string
	for _, c := range frozen.Cases {
		if strings.HasPrefix(c.ID, "SenseNova-") || strings.HasPrefix(c.ID, "Un-0/image-gen/") {
			expected = append(expected, c.ID)
		}
	}
	coverage := make([]mediaLatentBundleCase, len(bundle.Cases))
	for i, c := range bundle.Cases {
		coverage[i].Case = c.Case
	}
	if err := checkMediaLatentCoverage(coverage, expected); err != nil {
		t.Fatal(err)
	}
	if checkMediaLatentCoverage(coverage[1:], expected) == nil {
		t.Fatal("accepted omitted image case")
	}
	reused := 0
	for _, c := range bundle.Cases {
		checkTest(c.Test)
		index := slices.IndexFunc(frozen.Cases, func(v frozenCase) bool { return v.ID == c.Case })
		if index < 0 {
			t.Fatal("unexpected image")
		}
		reference := frozen.Cases[index]
		run, err := runrecord.RequireExactRun(t.Context(), store, c.Run)
		if err != nil {
			t.Fatal(err)
		}
		if run.Recipe != reference.Recipe || !slices.Equal(run.Inputs, reference.Inputs) || !slices.Equal(run.Outputs, []artifact.ID{c.Output}) {
			t.Fatal("frozen recipe or request differs")
		}
		output, found, err := artifact.ReadContent(t.Context(), store, c.Output)
		if err != nil || !found {
			t.Fatal("missing image", err)
		}
		observation := reference.Observations[0]
		observation.Artifact = c.Output
		if err := checkImageVideoObservation(observation, output); err != nil {
			t.Fatal(err)
		}
		switch c.Mode {
		case "acquired-cuda", "acquired-host":
			acquisition := read(c.Acquisition)
			var header struct {
				Source   string `json:"source_base"`
				Protocol string `json:"protocol_sha256"`
				Backend  string `json:"backend"`
			}
			if err := json.Unmarshal(acquisition, &header); err != nil {
				t.Fatal(err)
			}
			if header.Source != bundle.Source || header.Protocol != fmt.Sprintf("%x", sha256.Sum256(primary)) {
				t.Fatal("raw acquisition identity differs")
			}
			rows, closed, err := mediaLatentRows(acquisition)
			if err != nil {
				t.Fatal(err)
			}
			i := slices.IndexFunc(rows, func(row mediaLatentObservation) bool { return row.Case == c.Case })
			if i < 0 {
				t.Fatal("missing raw case")
			}
			row := rows[i]
			if row.Model != reference.Model || row.Input != reference.Inputs[0] || row.Output != c.Output {
				t.Fatal("raw model/request/output differs")
			}
			environment, err := runrecord.RequireEnvironment(t.Context(), store, row.Environment)
			if err != nil {
				t.Fatal(err)
			}
			check := func(candidate mediaLatentObservation) error {
				if c.Mode == "acquired-cuda" {
					return checkMediaLatentObservation(candidate, run, bundle.Source, closed)
				}
				return checkMediaCPUImage(candidate, run, bundle.Source)
			}
			if err := check(row); err != nil {
				t.Fatal(err)
			}
			if c.Mode == "acquired-host" && (header.Backend != "host" || environment.Backend != "host" || row.Before != nil || row.Load != nil || row.Memory.PeakBytes != 0) {
				t.Fatal("CPU image mislabeled as GPU acquisition")
			}
			if c.Mode == "acquired-cuda" && environment.Backend != "cuda" {
				t.Fatal("wrong GPU environment")
			}
			bad := row
			bad.Finite = false
			if check(bad) == nil {
				t.Fatal("accepted nonfinite output")
			}
			bad = row
			bad.Input = artifact.ID{}
			if check(bad) == nil {
				t.Fatal("accepted changed request")
			}
		case "reused-source-compatible":
			reused++
			if c.Case != "SenseNova-U1-8B-MoT-Infographic-V3/image-gen/2" || c.Run != reference.Run || !slices.Equal(run.Outputs, reference.Outputs) || run.CodeCommit != "" || run.Environment.Valid() {
				t.Fatal("historical full case relabeled as a new run")
			}
			var oracle struct {
				Source struct {
					Prompt string `json:"prompt_sha256"`
				} `json:"source"`
				Request map[string]any `json:"request"`
				Quality struct {
					PNG string `json:"reviewed_png_sha256"`
				} `json:"overgo_quality"`
			}
			oracleRaw := read(c.Acquisition)
			if err := json.Unmarshal(oracleRaw, &oracle); err != nil {
				t.Fatal(err)
			}
			var oracleValue any
			if err := json.Unmarshal(oracleRaw, &oracleValue); err != nil {
				t.Fatal(err)
			}
			canonicalOracle, err := json.Marshal(oracleValue)
			if err != nil {
				t.Fatal(err)
			}
			currentOracle, err := os.ReadFile(filepath.Join(root, "fixtures/sensenova/full_generation_quality.json"))
			if err != nil {
				t.Fatal(err)
			}
			if err := checkMediaProtocolIdentity(currentOracle, fmt.Sprintf("%x", sha256.Sum256(canonicalOracle))); err != nil {
				t.Fatal(err)
			}
			var request map[string]any
			if err := json.Unmarshal(read(reference.Inputs[0]), &request); err != nil {
				t.Fatal(err)
			}
			prompt, ok := request["prompt"].(string)
			if !ok || fmt.Sprintf("%x", sha256.Sum256([]byte(prompt))) != oracle.Source.Prompt {
				t.Fatal("reused prompt differs")
			}
			delete(request, "prompt")
			a, _ := json.Marshal(request)
			b, _ := json.Marshal(oracle.Request)
			if string(a) != string(b) || fmt.Sprintf("%x", sha256.Sum256(output.Data)) != oracle.Quality.PNG {
				t.Fatal("reused native request or reviewed output differs")
			}
		default:
			t.Fatal("unknown provenance mode")
		}
		var review struct {
			Output       artifact.ID `json:"output"`
			Subject      string      `json:"subject"`
			Preservation string      `json:"source_preservation"`
			Defects      string      `json:"visible_defects"`
			Video        string      `json:"video"`
		}
		if err := json.Unmarshal(read(c.Review), &review); err != nil {
			t.Fatal(err)
		}
		if review.Output != c.Output || review.Subject == "" || review.Preservation == "" || review.Defects == "" || review.Video == "" {
			t.Fatal("incomplete image-specific review")
		}
	}
	if reused != 1 {
		t.Fatal("reused full case missing")
	}
}
