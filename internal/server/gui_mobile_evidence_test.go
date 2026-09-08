package server

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"testing/fstest"

	"overgo/internal/jsonfile"
	"overgo/internal/testevidence"
)

type guiMobileCapture struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}
type guiMobileCase struct {
	Passed      bool             `json:"passed"`
	Observation string           `json:"observation"`
	Capture     guiMobileCapture `json:"capture"`
}
type guiMobileDevice struct {
	Platform            string                   `json:"platform"`
	Physical            bool                     `json:"physical"`
	Device              string                   `json:"device"`
	OSVersion           string                   `json:"os_version"`
	BrowserVersion      string                   `json:"browser_version"`
	AssistiveTechnology string                   `json:"assistive_technology"`
	Cases               map[string]guiMobileCase `json:"cases"`
}
type guiMobileEvidence struct {
	UISHA256 string            `json:"ui_sha256"`
	Devices  []guiMobileDevice `json:"devices"`
}

var guiMobileCases = []string{"keyboard", "orientation", "safe-areas", "zoom", "paste-upload", "long-reply", "stop-recovery", "drawer-focus", "settings-focus", "screen-reader"}

// Hash the actual embedded GUI, so plan/evidence commits do not invalidate
// unchanged assets and any subsequent GUI edit requires fresh device evidence.
func guiMobileSource() (string, error) {
	digest := sha256.New()
	err := fs.WalkDir(webuiFS, ".", func(path string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return err
		}
		data, err := fs.ReadFile(webuiFS, path)
		if err != nil {
			return err
		}
		fmt.Fprintf(digest, "%s:%d:", path, len(data))
		digest.Write(data)
		return nil
	})
	return hex.EncodeToString(digest.Sum(nil)), err
}

func validateGUIMobileEvidence(value guiMobileEvidence, captures fs.FS, source string) error {
	if value.UISHA256 != source {
		return errors.New("mobile evidence: GUI source differs; repeat device acceptance on these assets")
	}
	platforms := map[string]string{"ios-safari": "VoiceOver", "android-chrome": "TalkBack"}
	seen := map[string]bool{}
	for _, device := range value.Devices {
		technology, known := platforms[device.Platform]
		if !known || seen[device.Platform] || !device.Physical || strings.TrimSpace(device.Device) == "" || strings.TrimSpace(device.OSVersion) == "" || strings.TrimSpace(device.BrowserVersion) == "" || !strings.Contains(device.AssistiveTechnology, technology) {
			return fmt.Errorf("mobile evidence: incomplete or duplicate physical device %q", device.Platform)
		}
		seen[device.Platform] = true
		for _, name := range guiMobileCases {
			result, found := device.Cases[name]
			if !found || !result.Passed || strings.TrimSpace(result.Observation) == "" || !fs.ValidPath(result.Capture.Path) {
				return fmt.Errorf("mobile evidence: %s/%s needs a passing observation and capture", device.Platform, name)
			}
			data, err := fs.ReadFile(captures, result.Capture.Path)
			if err != nil {
				return fmt.Errorf("mobile evidence: %s/%s capture: %w", device.Platform, name, err)
			}
			digest := sha256.Sum256(data)
			if result.Capture.SHA256 != hex.EncodeToString(digest[:]) {
				return fmt.Errorf("mobile evidence: %s/%s capture identity differs", device.Platform, name)
			}
			dimensions, _, err := image.DecodeConfig(bytes.NewReader(data))
			if err != nil || dimensions.Width <= 0 || dimensions.Height <= 0 {
				return fmt.Errorf("mobile evidence: %s/%s capture must be a valid PNG or JPEG", device.Platform, name)
			}
		}
	}
	if len(seen) != len(platforms) {
		return errors.New("mobile evidence: both physical iOS Safari and Android Chrome records are required")
	}
	return nil
}

// This operator-only test is deliberately outside the default browser prefix.
// It validates submitted evidence, not the truth of an unobserved device run.
func TestGUIMobileDeviceEvidence(t *testing.T) {
	source, err := guiMobileSource()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("GUI asset SHA-256: %s", source)
	file := os.Getenv("OVERGO_GUI_MOBILE_EVIDENCE")
	if file == "" {
		if testing.Short() {
			t.Skip(testevidence.ShortIntegrationSkip + ": physical mobile evidence belongs to the explicit operator signoff")
		}
		t.Fatal("physical mobile evidence not supplied; set OVERGO_GUI_MOBILE_EVIDENCE to the operator record")
	}
	var value guiMobileEvidence
	if err := jsonfile.DecodeStrict(file, &value); err != nil {
		t.Fatal(err)
	}
	if err := validateGUIMobileEvidence(value, os.DirFS(filepath.Dir(file)), source); err != nil {
		t.Fatal(err)
	}
	t.Logf("physical mobile evidence leg: validated %d source-bound device records and %d case results with capture hashes; operator observations supplied, physical actions not replayed by this test", len(value.Devices), len(value.Devices)*len(guiMobileCases))
}

func TestGUIMobileEvidenceContract(t *testing.T) {
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	capture := encoded.Bytes()
	digest := sha256.Sum256(capture)
	source, err := guiMobileSource()
	if err != nil {
		t.Fatal(err)
	}
	fixture := func() guiMobileEvidence {
		result := guiMobileEvidence{UISHA256: source}
		for platform, technology := range map[string]string{"ios-safari": "VoiceOver", "android-chrome": "TalkBack"} {
			device := guiMobileDevice{Platform: platform, Physical: true, Device: "synthetic validator fixture", OSVersion: "fixture", BrowserVersion: "fixture", AssistiveTechnology: technology, Cases: map[string]guiMobileCase{}}
			for _, name := range guiMobileCases {
				device.Cases[name] = guiMobileCase{Passed: true, Observation: "synthetic validation only", Capture: guiMobileCapture{Path: "capture.png", SHA256: hex.EncodeToString(digest[:])}}
			}
			result.Devices = append(result.Devices, device)
		}
		return result
	}
	captures := fstest.MapFS{"capture.png": {Data: capture}}
	if err := validateGUIMobileEvidence(fixture(), captures, source); err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name   string
		change func(*guiMobileEvidence)
	}{
		{"stale source", func(value *guiMobileEvidence) { value.UISHA256 = "stale" }},
		{"missing platform", func(value *guiMobileEvidence) { value.Devices = value.Devices[:1] }},
		{"emulation", func(value *guiMobileEvidence) { value.Devices[0].Physical = false }},
		{"missing version", func(value *guiMobileEvidence) { value.Devices[0].BrowserVersion = "" }},
		{"missing screen reader", func(value *guiMobileEvidence) { value.Devices[0].AssistiveTechnology = "" }},
		{"missing case", func(value *guiMobileEvidence) { delete(value.Devices[0].Cases, "keyboard") }},
		{"failed case", func(value *guiMobileEvidence) {
			result := value.Devices[0].Cases["keyboard"]
			result.Passed = false
			value.Devices[0].Cases["keyboard"] = result
		}},
		{"missing observation", func(value *guiMobileEvidence) {
			result := value.Devices[0].Cases["keyboard"]
			result.Observation = ""
			value.Devices[0].Cases["keyboard"] = result
		}},
		{"capture outside record", func(value *guiMobileEvidence) {
			result := value.Devices[0].Cases["keyboard"]
			result.Capture.Path = "../capture.png"
			value.Devices[0].Cases["keyboard"] = result
		}},
		{"capture changed", func(value *guiMobileEvidence) {
			result := value.Devices[0].Cases["keyboard"]
			result.Capture.SHA256 = "wrong"
			value.Devices[0].Cases["keyboard"] = result
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			value := fixture()
			test.change(&value)
			if err := validateGUIMobileEvidence(value, captures, source); err == nil {
				t.Fatal("incomplete evidence accepted")
			}
		})
	}
	if err := validateGUIMobileEvidence(fixture(), fstest.MapFS{}, source); err == nil {
		t.Fatal("absent capture accepted")
	}
}
