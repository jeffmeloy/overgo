package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"image/gif"
	"os"
	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

const imageVideoWanSHA256 = "9eb2557509af96b4180b5364ba71da4103d393dbaf01d5abcef75f7cf5280183"

type mediaWanBundle struct {
	Version         uint16                 `json:"version"`
	Source          string                 `json:"source_base"`
	Previous        string                 `json:"previous_source"`
	Protocol        artifact.ID            `json:"protocol"`
	Cases           []mediaRoutedCase      `json:"cases"`
	Numerical       []mediaSharedCheck     `json:"numerical"`
	Instrumentation map[string]artifact.ID `json:"instrumentation"`
	Conditioning    artifact.ID            `json:"conditioning"`
	Inspection      artifact.ID            `json:"full_clip_inspection"`
	Frames          []artifact.ID          `json:"representative_frames"`
	Reuse           artifact.ID            `json:"reuse_binding"`
	Scope           string                 `json:"scope"`
}

// Compare complete decoded GIFs so frame order, pixels, palette and disposal
// remain checked independently of the recorded review. GIF time uses 100 ticks
// per second; cumulative rounding permits at most half a tick of total error.
func checkMediaWanGIF(current, historical []byte, observation imageVideoObservation) error {
	actual, err := gif.DecodeAll(bytes.NewReader(current))
	if err != nil {
		return err
	}
	old, err := gif.DecodeAll(bytes.NewReader(historical))
	if err != nil {
		return err
	}
	delay := 0
	for _, ticks := range actual.Delay {
		if ticks <= 0 {
			return errors.New("nonpositive frame delay")
		}
		delay += ticks
	}
	if len(actual.Image) != observation.Frames || actual.Config.Width != observation.Width || actual.Config.Height != observation.Height || observation.FPS <= 0 {
		return errors.New("video geometry or frame count differs")
	}
	delta := delay*observation.FPS - observation.Frames*100
	if delta < 0 {
		delta = -delta
	}
	if delta*2 > observation.FPS {
		return errors.New("video duration exceeds quantization bound")
	}
	old.Delay = actual.Delay
	if !reflect.DeepEqual(old, actual) {
		return errors.New("video differs beyond corrected delay")
	}
	return nil
}

func TestImageVideoWanAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": retained Wan evidence")
	}
	root := testutil.RepoRoot(t)
	document, err := plan.Load(filepath.Join(root, plan.Path))
	if err != nil {
		t.Fatal(err)
	}
	if document.Lane != "image_video_gen" || os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: explicit media data root required")
	}
	var bundle mediaWanBundle
	path := filepath.Join(root, "docs/image_video_wan.json")
	if err := jsonfile.DecodeStrict(path, &bundle); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkMediaProtocolIdentity(raw, imageVideoWanSHA256); err != nil {
		t.Fatal(err)
	}
	if bundle.Version != artifact.InitialDocumentVersion || bundle.Scope == "" || len(bundle.Numerical) != 5 || len(bundle.Frames) != 4 {
		t.Fatal("incomplete Wan bundle")
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
			t.Fatal("missing Wan content", id, err)
		}
		return content.Data
	}
	decode := func(id artifact.ID, value any) {
		t.Helper()
		if err := json.Unmarshal(read(id), value); err != nil {
			t.Fatal(err)
		}
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
			t.Fatal("empty Wan check")
		}
	}
	var merged mediaMergedEvidence
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs/image_video_merged.json"), &merged); err != nil {
		t.Fatal(err)
	}
	for _, revision := range []string{bundle.Previous, bundle.Source, "HEAD"} {
		if err := checkMediaRuntimeAtRevision(root, revision, merged.RuntimePaths, merged.RuntimeSHA256); err != nil {
			t.Fatal(err)
		}
	}
	protocolRaw := read(bundle.Protocol)
	var header struct {
		Source  string `json:"source_base"`
		Quality string `json:"quality_protocol_sha256"`
	}
	if err := json.Unmarshal(protocolRaw, &header); err != nil {
		t.Fatal(err)
	}
	if header.Source != bundle.Source || header.Quality != imageVideoProtocolSHA256 {
		t.Fatal("acquisition protocol changed")
	}
	var instrumentation struct {
		Source   string `json:"source_base"`
		Protocol string `json:"protocol_sha256"`
		Harness  string `json:"harness_sha256"`
	}
	decode(bundle.Instrumentation["media-wan-small-acquisition-instrumentation.json"], &instrumentation)
	if instrumentation.Source != bundle.Source || instrumentation.Protocol != fmt.Sprintf("%x", sha256.Sum256(protocolRaw)) || instrumentation.Harness != fmt.Sprintf("%x", sha256.Sum256(read(bundle.Instrumentation["media_wan_frozen_acquisition_test.go"]))) {
		t.Fatal("acquisition instrumentation differs")
	}
	failed, err := testevidence.GoTestJSONReport(string(read(bundle.Instrumentation["media-wan-small-acquisition-failed-1.jsonl"])))
	if err != nil || len(failed.Failed) == 0 {
		t.Fatal("failed instrumentation attempt was not retained", err)
	}
	quality, err := os.ReadFile(filepath.Join(root, "docs/image_video_protocol.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := checkImageVideoProtocolIdentity(quality); err != nil {
		t.Fatal(err)
	}
	var protocol imageVideoProtocol
	if err := json.Unmarshal(quality, &protocol); err != nil {
		t.Fatal(err)
	}
	var conditioning []struct {
		Case          string      `json:"case"`
		Input         artifact.ID `json:"input"`
		Conditional   string      `json:"conditional_f32_sha256"`
		Unconditional string      `json:"unconditional_f32_sha256"`
		Prompt        string      `json:"reference_prompt"`
	}
	decode(bundle.Conditioning, &conditioning)
	seen := map[string]bool{}
	expected := 0
	for _, c := range protocol.Cases {
		if !strings.HasPrefix(c.ID, "Wan2.1-") {
			continue
		}
		expected++
		index := slices.IndexFunc(bundle.Cases, func(v mediaRoutedCase) bool { return v.Case == c.ID })
		if index < 0 || seen[c.ID] {
			t.Fatal("missing or duplicated Wan case")
		}
		seen[c.ID] = true
		entry := bundle.Cases[index]
		run, err := runrecord.RequireExactRun(t.Context(), store, entry.Run)
		if err != nil {
			t.Fatal(err)
		}
		if run.Recipe != c.Recipe || !slices.Equal(run.Inputs, c.Inputs) || !slices.Equal(run.Outputs, []artifact.ID{entry.Output}) || run.Outcome != runrecord.OutcomeSucceeded || run.MeasuredNS == 0 || len(run.Phases) == 0 {
			t.Fatal("Wan run identity or measurement missing")
		}
		if _, err := runrecord.RequireEnvironment(t.Context(), store, run.Environment); err != nil {
			t.Fatal(err)
		}
		if err := checkMediaWanGIF(read(entry.Output), read(c.Outputs[0]), c.Observations[0]); err != nil {
			t.Fatal(c.ID, err)
		}
		checkTest(entry.Test)
		var review struct {
			Case         string `json:"case"`
			SHA          string `json:"gif_sha256"`
			Subject      string `json:"subject"`
			Preservation string `json:"source_preservation"`
			Defects      string `json:"visible_defects"`
			Video        string `json:"video"`
			Scope        string `json:"review_scope"`
		}
		decode(entry.Review, &review)
		if review.Case != c.ID || review.SHA != fmt.Sprintf("%x", sha256.Sum256(read(entry.Output))) || review.Subject == "" || review.Preservation == "" || review.Defects == "" || review.Video == "" || review.Scope == "" {
			t.Fatal("incomplete content-bound visual review")
		}
		ci := slices.IndexFunc(conditioning, func(v struct {
			Case          string      `json:"case"`
			Input         artifact.ID `json:"input"`
			Conditional   string      `json:"conditional_f32_sha256"`
			Unconditional string      `json:"unconditional_f32_sha256"`
			Prompt        string      `json:"reference_prompt"`
		}) bool {
			return v.Case == c.ID
		})
		if ci < 0 || conditioning[ci].Input != c.Inputs[0] || conditioning[ci].Conditional == "" || conditioning[ci].Unconditional == "" || conditioning[ci].Prompt == "" {
			t.Fatal("missing prompt provenance")
		}
		switch entry.Mode {
		case "acquired-cuda":
			var group struct {
				Loads  int                           `json:"model_loads"`
				Load   uint64                        `json:"load_ns"`
				Close  uint64                        `json:"close_ns"`
				Closed map[string]driver.MemoryStats `json:"closed"`
				Rows   []struct {
					Case      string                `json:"case"`
					Run       artifact.ID           `json:"validation_run"`
					Workflow  artifact.ID           `json:"workflow_run"`
					Wall      uint64                `json:"request_ns"`
					Allocated uint64                `json:"allocated_bytes"`
					Peak      uint64                `json:"peak_host_bytes"`
					Finite    bool                  `json:"prequantization_finite"`
					Same      bool                  `json:"identical_except_delay"`
					DenBefore driver.ExecutionStats `json:"denoiser_before"`
					DenAfter  driver.ExecutionStats `json:"denoiser_after"`
					DecBefore driver.ExecutionStats `json:"decoder_before"`
					DecAfter  driver.ExecutionStats `json:"decoder_after"`
				} `json:"observations"`
			}
			decode(entry.Acquisition, &group)
			if group.Loads != 1 || group.Load == 0 || group.Close == 0 || len(group.Rows) != 2 || len(group.Closed) != 2 {
				t.Fatal("incomplete acquired Wan resources")
			}
			for _, memory := range group.Closed {
				if memory.CurrentBytes != 0 || memory.PeakBytes == 0 {
					t.Fatal("Wan cleanup or memory measurement absent")
				}
			}
			found := false
			for _, row := range group.Rows {
				if row.Case != c.ID {
					continue
				}
				found = true
				if row.Run != run.ID || run.CodeCommit != bundle.Source || row.Wall != run.MeasuredNS || row.Allocated == 0 || row.Peak == 0 || !row.Finite || !row.Same || row.DenAfter.KernelLaunches+row.DenAfter.GraphLaunches <= row.DenBefore.KernelLaunches+row.DenBefore.GraphLaunches || row.DecAfter.KernelLaunches+row.DecAfter.GraphLaunches <= row.DecBefore.KernelLaunches+row.DecBefore.GraphLaunches || row.DecAfter.DeviceToHostBytes <= row.DecBefore.DeviceToHostBytes {
					t.Fatal("missing actual Wan execution")
				}
				workflow, err := runrecord.RequireExactRun(t.Context(), store, row.Workflow)
				if err != nil || workflow.Recipe != c.Recipe || !slices.Equal(workflow.Inputs, c.Inputs) || !slices.Equal(workflow.Outputs, run.Outputs) {
					t.Fatal("compiled workflow differs", err)
				}
			}
			if !found {
				t.Fatal("missing acquired Wan row")
			}
		case "reused-production-acquisition":
			var binding struct {
				Run         artifact.ID `json:"run"`
				Output      artifact.ID `json:"output"`
				Environment artifact.ID `json:"environment"`
				Acquisition artifact.ID `json:"acquisition"`
				Source      string      `json:"source_base"`
				Current     string      `json:"current_source"`
				Runtime     string      `json:"runtime_sha256"`
			}
			decode(bundle.Reuse, &binding)
			if binding.Run != run.ID || binding.Output != entry.Output || binding.Environment != run.Environment || binding.Acquisition != entry.Acquisition || binding.Source != bundle.Previous || binding.Current != bundle.Source || binding.Runtime != merged.RuntimeSHA256 {
				t.Fatal("full acquisition publication binding differs")
			}
			var inspection struct {
				Frames  int    `json:"frames"`
				Same    bool   `json:"identical_except_delay"`
				Changed int    `json:"changed_frames_from_historical"`
				SHA     string `json:"gif_sha256"`
				Rows    []struct {
					Index   int    `json:"index"`
					SHA     string `json:"rgba_sha256"`
					Changed int    `json:"changed_channels_from_historical"`
				} `json:"frame_observations"`
			}
			decode(bundle.Inspection, &inspection)
			if inspection.Frames != c.Observations[0].Frames || !inspection.Same || inspection.Changed != 0 || inspection.SHA != review.SHA || len(inspection.Rows) != inspection.Frames {
				t.Fatal("incomplete full-frame inspection")
			}
			for i, row := range inspection.Rows {
				if row.Index != i || row.SHA == "" || row.Changed != 0 {
					t.Fatal("frame inspection reordered or changed")
				}
			}
			var retained struct {
				Source    string                        `json:"source_base"`
				Input     string                        `json:"input_sha256"`
				Observed  string                        `json:"observed_at"`
				Completed string                        `json:"completed_at"`
				Closed    map[string]driver.MemoryStats `json:"after_close"`
				Rows      []struct {
					Wall     uint64   `json:"wall_ns"`
					Frames   int      `json:"frames"`
					SHA      string   `json:"gif_sha256"`
					Latent   string   `json:"latent_sha256"`
					FrameSHA []string `json:"frame_sha256"`
				} `json:"observations"`
			}
			decode(entry.Acquisition, &retained)
			if retained.Source != bundle.Previous || run.CodeCommit != bundle.Previous || retained.Input != strings.TrimPrefix(c.Inputs[0].String(), "file:sha256:") || retained.Observed == "" || retained.Completed == "" || len(retained.Rows) != 2 || retained.Rows[0].Wall != run.MeasuredNS {
				t.Fatal("retained acquisition rebound incorrectly")
			}
			for _, row := range retained.Rows {
				if row.Frames != c.Observations[0].Frames || row.SHA != review.SHA || row.Latent == "" || len(row.FrameSHA) != row.Frames {
					t.Fatal("truncated retained trajectory")
				}
			}
			if len(retained.Closed) != 2 {
				t.Fatal("missing retained cleanup")
			}
			for _, m := range retained.Closed {
				if m.CurrentBytes != 0 || m.PeakBytes == 0 {
					t.Fatal("retained cleanup differs")
				}
			}
		default:
			t.Fatal("unsupported Wan acquisition mode")
		}
	}
	if expected != len(bundle.Cases) || len(seen) != expected || len(conditioning) != expected {
		t.Fatal("Wan coverage differs")
	}
	for _, check := range bundle.Numerical {
		if check.Name == "" || check.Command == "" {
			t.Fatal("unnamed native check")
		}
		checkTest(check.Evidence)
	}
	for _, id := range bundle.Frames {
		if len(read(id)) == 0 {
			t.Fatal("missing representative frame")
		}
	}
	if len(read(bundle.Inspection)) == 0 || len(read(bundle.Reuse)) == 0 {
		t.Fatal("missing reuse inspection")
	}
}

func TestImageVideoWanGIFRejectsAlteredSequence(t *testing.T) {
	root := testutil.RepoRoot(t)
	var bundle mediaWanBundle
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs/image_video_wan.json"), &bundle); err != nil {
		t.Skip("integration: Wan bundle not present")
	}
	if os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: explicit media store required")
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
	content, found, err := artifact.ReadContent(t.Context(), store, bundle.Cases[0].Output)
	if err != nil || !found {
		t.Fatal(err)
	}
	original, err := gif.DecodeAll(bytes.NewReader(content.Data))
	if err != nil {
		t.Fatal(err)
	}
	observation := imageVideoObservation{Width: original.Config.Width, Height: original.Config.Height, Frames: len(original.Image), FPS: 16}
	for _, mutation := range []string{"truncate", "reorder", "timing", "pixel"} {
		t.Run(mutation, func(t *testing.T) {
			video, err := gif.DecodeAll(bytes.NewReader(content.Data))
			if err != nil {
				t.Fatal(err)
			}
			switch mutation {
			case "truncate":
				video.Image = video.Image[:len(video.Image)-1]
				video.Delay = video.Delay[:len(video.Delay)-1]
				if len(video.Disposal) > 0 {
					video.Disposal = video.Disposal[:len(video.Disposal)-1]
				}
			case "reorder":
				slices.Reverse(video.Image)
			case "timing":
				video.Delay[0] += 100
			case "pixel":
				video.Image[0].Pix[0] = (video.Image[0].Pix[0] + 1) % uint8(len(video.Image[0].Palette)-1)
			}
			var encoded bytes.Buffer
			if err := gif.EncodeAll(&encoded, video); err != nil {
				t.Fatal(err)
			}
			if err := checkMediaWanGIF(encoded.Bytes(), content.Data, observation); err == nil {
				t.Fatal("altered clip accepted")
			}
		})
	}
}
