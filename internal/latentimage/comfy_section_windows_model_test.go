//go:build modeltest && windows

package latentimage

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/safetensors"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// comfySection is a recorded run of a MOSS Ultra section in ComfyUI (comfy-recorder): the inputs set over its request,
// the sigmas each sampler ran, its nodes' texts, its saved files, and its request's nodes' inputs; and the run's time,
// the card's peak and each sampler's progress stamps, its step and the microseconds since the run began.
type comfySection struct {
	Set           map[string]any       `json:"set"`
	Sigmas        map[string][]float64 `json:"sigmas"`
	Texts         map[string][]string  `json:"texts"`
	Milliseconds  int64                `json:"milliseconds"`
	PeakCardBytes uint64               `json:"peak_card_bytes"`
	Progress      map[string][]struct {
		Value        int   `json:"value"`
		Microseconds int64 `json:"us"`
	} `json:"progress"`
	Files []struct {
		Node   string `json:"node"`
		Name   string `json:"name"`
		SHA256 string `json:"sha256"`
	} `json:"files"`
	request map[string]struct {
		Inputs map[string]any `json:"inputs"`
	}
	// folder is where the run's files lie beside the store, under references/comfy/moss-ultra.
	folder string
}

// ultraPicture is the SHA-256 of the picture the recorder gave the image sections, kept beside the store.
const ultraPicture = "8f6948d0518656d9d19c2dc797e05e95fd990e37d84eadf0436d52be9de53d8a"

// readComfySection reads the recording of section (its slug) from run, and its section's request.
func readComfySection(t *testing.T, run, slug string) comfySection {
	t.Helper()
	repo := testutil.RepoRoot(t)
	var section comfySection
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join(repo, "fixtures", "comfy", "moss-ultra", "recordings", run, slug+".json")))(t), &section))
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join(repo, "fixtures", "comfy", "moss-ultra", "requests", slug+".json")))(t), &section.request))
	// The inputs the recorder set over the request are the ones ComfyUI ran (SQA-457): a case reading the request's own
	// had read the template's seed and an empty builder where the recording set others.
	for at, value := range section.Set {
		node, input, named := strings.Cut(at, ".")
		if _, held := section.request[node].Inputs[input]; !named || !held {
			t.Fatalf("the recording sets %s, which names no input of the section's request", at)
		}
		section.request[node].Inputs[input] = value
	}
	section.folder = filepath.Join(testcheck.Must(dataroot.Resolve(repo))(t).References(), "comfy", "moss-ultra", run, slug)
	return section
}

// inputs are a node's inputs as the section's request gives them.
func (s comfySection) inputs(node string) map[string]any { return s.request[node].Inputs }

// latent is the latent the recorder saved after node, [1 128 36 64], by the SHA-256 the recording names.
func (s comfySection) latent(t *testing.T, node string) []float32 {
	t.Helper()
	for _, file := range s.Files {
		if file.Node != latentNode(node) {
			continue
		}
		path := filepath.Join(s.folder, file.Name)
		if sum := sha256.Sum256(testcheck.Must(os.ReadFile(path))(t)); hex.EncodeToString(sum[:]) != file.SHA256 {
			t.Fatalf("%s hashes %x; want the recorded %s", path, sum, file.SHA256)
		}
		source := testcheck.Must(safetensors.OpenFile(path))(t)
		defer source.Close()
		value := source.Tensors["latent_tensor"]
		if len(value.Shape) != 4 || value.Shape[0] != 1 || value.Shape[1] != 128 || value.Shape[2] != 36 || value.Shape[3] != 64 {
			t.Fatalf("%s holds a latent of shape %v; want [1 128 36 64]", path, value.Shape)
		}
		return testcheck.Must(safetensors.ReadF32(value))(t)
	}
	t.Fatalf("the recording keeps no latent of %s", node)
	return nil
}

// ultraPictureBytes is the picture the recorder gave the image sections, by its SHA-256.
func ultraPictureBytes(t *testing.T) []byte {
	t.Helper()
	path := filepath.Join(testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t).References(), "comfy", "moss-ultra", "overgo-recorder-input.png")
	picture := testcheck.Must(os.ReadFile(path))(t)
	if sum := sha256.Sum256(picture); hex.EncodeToString(sum[:]) != ultraPicture {
		t.Fatalf("the recorder's picture hashes %x; want %s", sum, ultraPicture)
	}
	return picture
}

// latentNode is the node a recording names a saved latent by: the saver the recorder adds after the node.
func latentNode(node string) string { return "overgo-latent-" + node }
