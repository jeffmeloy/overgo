package server

import (
	"bytes"
	"encoding/binary"
	"net/http"
	"net/http/httptest"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"strconv"
	"strings"
	"testing"
)

func TestTranscriptionCaptureDeclaration(t *testing.T) {
	fixture := newTranscriptionHTTPFixture(t, nil)
	caps, err := fixture.workspace.WorkflowCapabilities(t.Context(), WorkflowGeneration)
	if err != nil || len(caps) != 1 {
		t.Fatalf("capabilities: %v %v", caps, err)
	}
	control := caps[0].Controls[0]
	base, _, err := modelrecipe.SpeechComponents(fixture.workspace.program.Definition())
	if err != nil {
		t.Fatal(err)
	}
	id, _ := base.PrimaryDependency(recipe.DependencyProfile)
	contract, err := modelrecipe.RequireAudioContract(t.Context(), fixture.workspace.store, id)
	if err != nil {
		t.Fatal(err)
	}
	if control.Audio == nil || control.Audio.Format != contract.Format || control.Audio.MaximumSamples != fixture.workspace.policy.Inspection.MaximumSamples || control.Audio.MaximumEncodedBytes != fixture.workspace.policy.Inspection.MaximumEncodedBytes {
		t.Fatalf("capture differs from recipe admission: %+v", control.Audio)
	}
	if err := validateWorkflowCapabilities(caps); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"rate", "bytes", "samples", "type", "media"} {
		t.Run(name, func(t *testing.T) {
			changed := caps[0]
			changed.Controls = append([]WorkflowControl(nil), changed.Controls...)
			audio := *control.Audio
			changed.Controls[0].Audio = &audio
			switch name {
			case "rate":
				audio.Format.SampleRate = 0
			case "bytes":
				audio.MaximumEncodedBytes = 0
			case "samples":
				audio.MaximumSamples = 0
			case "type":
				changed.Controls[0].Type = WorkflowControlText
			case "media":
				changed.Controls[0].Media = "image"
			}
			if err := validateWorkflowCapabilities([]WorkflowCapability{changed}); err == nil {
				t.Fatal("invalid capture declaration accepted")
			}
		})
	}
}

func TestNativeAudioFormatRefusal(t *testing.T) {
	fixture := newTranscriptionHTTPFixture(t, nil)
	wave := bytes.Clone(fixture.wave)
	rate := binary.LittleEndian.Uint32(wave[24:28])
	binary.LittleEndian.PutUint32(wave[24:28], 2*rate)
	binary.LittleEndian.PutUint32(wave[28:32], 2*binary.LittleEndian.Uint32(wave[28:32]))
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, fixture.request(t, wave, nil))
	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), strconv.FormatUint(uint64(rate), 10)+" Hz") || strings.Contains(response.Body.String(), "admission policy") {
		t.Fatalf("format refusal: HTTP %d %s", response.Code, response.Body.String())
	}
	// Changing the rate must not convert or activate the mismatched upload.
	statuses := fixture.handler.operations.List()
	if len(statuses) != 1 || len(statuses[0].Outputs) != 0 {
		t.Fatalf("mismatched audio produced output: %+v", statuses)
	}
}
