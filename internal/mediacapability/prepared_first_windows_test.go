//go:build windows

package mediacapability

import (
	"context"
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/latentimage"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodbtest"
	"overgo/internal/strictjson"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAPresetOnlyRequestTakesItsPresetsSize holds SQA-409: a MOSS request naming a section preset and no size, as recipe
// run takes it, reaches the picture session with the size the preset's canvas gives, its preset still named for the
// rest, and passes the session's check of its size; a request stating its size reaches it as it came; and a request's
// own canvas gives its size in place of the canvas.
func TestAPresetOnlyRequestTakesItsPresetsSize(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	testcheck.NoError(t, os.WriteFile(filepath.Join(directory, "model_index.json"), []byte(`{"_class_name": "MOSSImagePipeline"}`), 0o600))
	var reached []string
	session := func(_ context.Context, _ artifact.Repository, _ string, _ modelrecipe.CapabilityEvidenceSelection, input string) (any, error) {
		reached = append(reached, input)
		return "drawn", nil
	}
	run := func(input string) latentimage.Request {
		t.Helper()
		testcheck.Must(preparedFirst(session)(t.Context(), nil, directory, modelrecipe.CapabilityEvidenceSelection{}, input))(t)
		var taken latentimage.Request
		testcheck.NoError(t, strictjson.DecodeBytes([]byte(reached[len(reached)-1]), &taken))
		return taken
	}
	canvas := testcheck.Must(func() ([2]int, error) {
		width, height, err := media.Canvas{Aspect: "16:9", Megapixels: 3, Multiple: 32}.Size()
		return [2]int{width, height}, err
	}())(t)
	if taken := run(`{"prompt":"a fox","preset":"ultra-text-to-image","seed":42}`); [2]int{taken.Width, taken.Height} != canvas ||
		taken.Preset != "ultra-text-to-image" || taken.Seed != 42 || taken.Canvas != nil || latentimage.ValidateRequest(taken) != nil {
		t.Errorf("a preset-only request reached the session as %+v (%v); want the preset's canvas, %v, and the preset named", taken, latentimage.ValidateRequest(taken), canvas)
	}
	own := `{"prompt":"a fox","preset":"ultra-text-to-image","width":1024,"height":576}`
	if run(own); reached[len(reached)-1] != own {
		t.Errorf("a request stating its size reached the session as %s", reached[len(reached)-1])
	}
	if taken := run(`{"prompt":"a fox","canvas":{"aspect":"1:1","megapixels":1,"multiple":64}}`); taken.Width != 1024 || taken.Height != 1024 || taken.Canvas != nil {
		t.Errorf("a request's own canvas reached the session as %+v; want its size, 1024 x 1024, in its place", taken)
	}
}

// TestAStoredPictureReachesItsSessionAsItsBytes holds the pictures a page attaches to a MOSS request
// (moss-ultra-compare/stored-pictures): a request naming its picture and its depth map by their stored identity reaches
// the picture session with each one's bytes in base64 and the rest as it asked; one giving its picture as its bytes
// reaches it as it came; one naming a picture the store does not hold, or a stored record that is no file, is refused
// before the session; and a rotary page offers the picture as a slot of image media beside the denoise and the enhance,
// where a flow profile's page offers none of them.
func TestAStoredPictureReachesItsSessionAsItsBytes(t *testing.T) {
	store := overgodbtest.Open(t, t.TempDir())
	stored := func(data []byte) artifact.Content {
		id := testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, data))(t)
		return artifact.Content{Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(data)), MediaType: "image/png"}, Data: data}
	}
	picture, depth := stored([]byte("a picture")), stored([]byte("a depth map"))
	testcheck.Must(artifact.Publish(t.Context(), store, artifact.Batch{Key: "fixture/stored-pictures",
		Artifacts: []artifact.Descriptor{picture.Descriptor, depth.Descriptor}, Contents: []artifact.Content{picture, depth}}))(t)
	var reached []string
	session := func(_ context.Context, _ artifact.Repository, _ string, _ modelrecipe.CapabilityEvidenceSelection, input string) (any, error) {
		reached = append(reached, input)
		return "drawn", nil
	}
	run := preparedFirst(session)
	named := `{"prompt":"a fox","width":1024,"height":576,"image":"` + picture.Descriptor.ID.String() + `","denoise":0.72,"control":{"map":"` + depth.Descriptor.ID.String() + `","start":0.1},"adapter":"moss-control"}`
	if got := testcheck.Must(run(t.Context(), store, "", modelrecipe.CapabilityEvidenceSelection{}, named))(t); got != "drawn" || len(reached) != 1 {
		t.Fatalf("the request naming stored pictures gave %v and reached the session %d times", got, len(reached))
	}
	var taken latentimage.Request
	testcheck.NoError(t, strictjson.DecodeBytes([]byte(reached[0]), &taken))
	if taken.Image != base64.StdEncoding.EncodeToString(picture.Data) || taken.Control == nil || taken.Control.Map != base64.StdEncoding.EncodeToString(depth.Data) ||
		taken.Control.Start != 0.1 || taken.Prompt != "a fox" || taken.Width != 1024 || taken.Denoise != 0.72 {
		t.Errorf("the session took %s; want the stored picture's and map's bytes and the rest as asked", reached[0])
	}
	bytes := `{"prompt":"a fox","width":1024,"height":576,"image":"` + base64.StdEncoding.EncodeToString(picture.Data) + `"}`
	testcheck.Must(run(t.Context(), store, "", modelrecipe.CapabilityEvidenceSelection{}, bytes))(t)
	if reached[len(reached)-1] != bytes {
		t.Errorf("a picture given as its bytes reached the session as %s", reached[len(reached)-1])
	}
	absent := testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, []byte("never stored")))(t)
	record := testcheck.Must(artifact.IdentifyBytes(artifact.KindEvaluation, []byte("a record")))(t)
	for _, refused := range []string{`{"prompt":"a fox","width":1024,"height":576,"image":"` + absent.String() + `"}`,
		`{"prompt":"a fox","width":1024,"height":576,"control":{"map":"` + record.String() + `"},"adapter":"moss-control"}`} {
		before := len(reached)
		if _, err := run(t.Context(), store, "", modelrecipe.CapabilityEvidenceSelection{}, refused); err == nil || len(reached) != before || !strings.Contains(err.Error(), "stored") {
			t.Errorf("%s was taken (%v) or reached the session; want it refused for its picture", refused, err)
		}
	}

	root := testutil.ExtractTarGzip(t, testutil.FixturePath(t, "moss", "generator-compact.tar.gz"), "150c4263057d72476ef4399a442c2b7dbadcb955670232fc2ea869c71f07c5db")
	controls, refusal := Controls(modelrecipe.ModuleLatentImagePrepare, root)
	if refusal != "" {
		t.Fatal(refusal)
	}
	slot := slices.IndexFunc(controls, func(c Control) bool { return c.Name == "image" })
	if slot < 0 || controls[slot].Type != ControlArtifact || controls[slot].Media != "image" || controls[slot].Label != "picture" || controls[slot].Required || controls[slot].Advanced {
		t.Errorf("MOSS's page offers its picture as %+v; want an optional slot of image media beside the prompt", controls)
	}
	for _, name := range []string{"denoise", "enhance"} {
		if !slices.ContainsFunc(controls, func(c Control) bool { return c.Name == name }) {
			t.Errorf("MOSS's page offers no %s", name)
		}
	}
	krea := t.TempDir()
	testutil.WriteImageProfileFixture(t, krea)
	controls, refusal = Controls(modelrecipe.ModuleLatentImagePrepare, krea)
	if refusal != "" {
		t.Fatal(refusal)
	}
	if slices.ContainsFunc(controls, func(c Control) bool { return slices.Contains(rotaryControls, c.Name) }) {
		t.Errorf("a flow profile's page offers a rotary profile's control: %+v", controls)
	}
}

// TestAPictureRequestAskingNoEnhancingGoesToItsSessionAsItIs holds the image capability's prompt mode 2 step
// (moss-ultra-sections/prompt-modes): a request asking no enhancing, an input that is no request, and a request asking
// enhancing that the session's strict reading refuses, by an unknown member or a repeated one (SQA-408), reach the
// picture session as they came, the session's answer returned, no chat opened (the store and model are none).
func TestAPictureRequestAskingNoEnhancingGoesToItsSessionAsItIs(t *testing.T) {
	var reached []string
	session := func(_ context.Context, _ artifact.Repository, _ string, _ modelrecipe.CapabilityEvidenceSelection, input string) (any, error) {
		reached = append(reached, input)
		return "drawn", nil
	}
	inputs := []string{`{"prompt":"a fox","width":1024,"height":576}`, `{"prompt":"a fox","width":1024,"height":576,"enhance":false}`, `not a request`,
		`{"enhance":true,"prompt":"a fox","width":1024,"height":576,"stepps":1}`, `{"enhance":true,"prompt":"a fox","steps":1,"steps":50}`}
	for _, input := range inputs {
		if got := testcheck.Must(preparedFirst(session)(t.Context(), nil, "", modelrecipe.CapabilityEvidenceSelection{}, input))(t); got != "drawn" {
			t.Errorf("%s gave %v, want the session's answer", input, got)
		}
	}
	for index, input := range inputs {
		if reached[index] != input {
			t.Errorf("the session took %q, want %q as it came", reached[index], input)
		}
	}
}

// TestAnInvalidRequestIsRefusedBeforeItsFilesAreRead holds SQA-425: a MOSS request with steps of -1, naming its adapter
// and its picture by its stored identity, is refused for its sampling policy, as its session would refuse it, whether
// the picture is stored or not, before the session and before any file it names is read: no file is named for the
// adapter's content (pictureFiles), where the same request with its steps admitted names the adapter's. A request
// asking prompt mode 2 beside a layout is refused as its enhancement refuses it.
func TestAnInvalidRequestIsRefusedBeforeItsFilesAreRead(t *testing.T) {
	store := overgodbtest.Open(t, t.TempDir())
	data := []byte("a picture")
	stored := artifact.Content{Descriptor: artifact.Descriptor{ID: testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, data))(t), Size: uint64(len(data)), MediaType: "image/png"}, Data: data}
	testcheck.Must(artifact.Publish(t.Context(), store, artifact.Batch{Key: "fixture/refused-picture", Artifacts: []artifact.Descriptor{stored.Descriptor}, Contents: []artifact.Content{stored}}))(t)
	absent := testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, []byte("never stored")))(t)
	reached := 0
	session := func(context.Context, artifact.Repository, string, modelrecipe.CapabilityEvidenceSelection, string) (any, error) {
		reached++
		return "drawn", nil
	}
	serve := capabilityruntime.Reading(preparedFirst(session), capabilityruntime.Contents(pictureFiles))
	root := t.TempDir()
	request := func(steps int, picture artifact.ID) string {
		return fmt.Sprintf(`{"prompt":"a fox","width":1024,"height":576,"steps":%d,"adapter":"moss-control","image":%q}`, steps, picture.String())
	}
	for _, picture := range []artifact.ID{stored.Descriptor.ID, absent} {
		if _, err := serve(t.Context(), store, root, modelrecipe.CapabilityEvidenceSelection{}, request(-1, picture)); err == nil || err.Error() != "latent image: invalid sampling policy" || reached != 0 {
			t.Errorf("a request with steps of -1 and its picture %s was refused with %v, the session reached %d times; want its sampling policy refused before the session", picture, err, reached)
		}
		if files := testcheck.Must(pictureFiles(root, request(-1, picture)))(t); len(files) != 0 {
			t.Errorf("a request with steps of -1 named %v for its content; want none read", files)
		}
	}
	if files := testcheck.Must(pictureFiles(root, request(8, stored.Descriptor.ID)))(t); len(files) != 1 || filepath.Base(files[0]) != "moss-control.safetensors" {
		t.Errorf("an admitted request named %v for its content; want its adapter", files)
	}
	layout := `{"enhance":true,"width":1024,"height":576,"layout":{"high_level_description":"a fox"}}`
	if _, err := serve(t.Context(), store, root, modelrecipe.CapabilityEvidenceSelection{}, layout); err == nil || reached != 0 || !strings.Contains(err.Error(), "prompt mode 2") {
		t.Errorf("a request asking prompt mode 2 beside a layout was refused with %v, the session reached %d times; want the enhancement's refusal", err, reached)
	}
}
