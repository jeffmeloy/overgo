package main

import (
	"bufio"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"overgo/internal/modelartifact"
	"overgo/internal/strictjson"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// fakeComfyUI answers as ComfyUI's HTTP API and websocket do for one prompt: its stats with the card's free memory
// falling and rising, the queue taking the prompt, a history with no entry and then the run's, each file the run wrote,
// the models it lists by folder, and over the websocket the run's start, its sampler's two steps, a preview and a ping,
// and its success. It keeps the request posted and counts the posts.
type fakeComfyUI struct {
	failed bool
	// interrupted answers the history with a server error, as a ComfyUI that stops answering mid-run.
	interrupted bool
	mu          sync.Mutex
	posted      map[string]apiNode
	posts       int
	reads       int
	free        []uint64
	// accepted is closed when the prompt is posted, so the websocket tells the run's events after it.
	accepted chan struct{}
	// quit is closed when the test ends, so a socket whose prompt was never posted stops waiting.
	quit chan struct{}
	// ponged is the payload of the pong the recorder answered the websocket's ping with.
	ponged []byte
}

func newFake(free ...uint64) *fakeComfyUI {
	return &fakeComfyUI{free: free, accepted: make(chan struct{}), quit: make(chan struct{})}
}

// answer writes value as a JSON answer.
func answer(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}

// served are the bytes of the files the fake's run wrote, by name.
var served = map[string]string{"original.png": "picture bytes", "latent.latent": "latent bytes"}

// listed are the model files the fake's ComfyUI lists, by folder.
var listed = map[string][]string{
	"diffusion_models": {"moss_image_base_int8_convrot.safetensors"},
	"text_encoders":    {"Qwen3.5-9B-abliterated_int8_convrot.safetensors"},
	"vae":              {"flux2-vae_small.safetensors"},
	"loras":            {"moss_image_sft.safetensors"},
}

// latentNode is the node a recording adds to save the sampler's latent, named as the command names it.
const latentNode = "overgo-latent-598"

// recorded is a recording as the command writes it, read here by its own fields, so the test builds where the command
// writes none.
type recorded struct {
	Workflow      string               `json:"workflow_sha256"`
	Section       string               `json:"section"`
	Request       string               `json:"request_sha256"`
	Set           map[string]any       `json:"set"`
	ComfyUI       string               `json:"comfyui_version"`
	Commits       map[string]string    `json:"commits"`
	Models        []recordedModel      `json:"models"`
	Prompt        string               `json:"prompt_id"`
	Milliseconds  int64                `json:"milliseconds"`
	PeakCardBytes uint64               `json:"peak_card_bytes"`
	Sigmas        map[string][]float64 `json:"sigmas"`
	Progress      map[string][]struct {
		Value        int   `json:"value"`
		Max          int   `json:"max"`
		Microseconds int64 `json:"us"`
	} `json:"progress"`
	Resumed bool                `json:"resumed"`
	Texts   map[string][]string `json:"texts"`
	Cached  []string            `json:"cached"`
	Files   []struct {
		Node, Kind, Type, Name, SHA256 string
		Bytes                          int
	} `json:"files"`
}

func (f *fakeComfyUI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/ws" {
		f.socket(w, r)
		return
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	switch {
	case r.URL.Path == "/system_stats":
		free := f.free[min(f.reads, len(f.free)-1)]
		f.reads++
		answer(w, map[string]any{"system": map[string]any{"comfyui_version": "0.3.fake"}, "devices": []any{map[string]any{"vram_total": 1000, "vram_free": free}}})
	case strings.HasPrefix(r.URL.Path, "/models/"):
		answer(w, append([]string{}, listed[strings.TrimPrefix(r.URL.Path, "/models/")]...))
	case r.URL.Path == "/prompt":
		var body struct {
			Prompt map[string]apiNode `json:"prompt"`
			Client string             `json:"client_id"`
		}
		if json.NewDecoder(r.Body).Decode(&body) != nil || body.Client != clientID {
			http.Error(w, "unreadable", http.StatusBadRequest)
			return
		}
		f.posted = body.Prompt
		f.posts++
		close(f.accepted)
		answer(w, map[string]any{"prompt_id": "p1", "number": 0, "node_errors": map[string]any{}})
	case r.URL.Path == "/history/p1" && f.interrupted:
		http.Error(w, "gone", http.StatusInternalServerError)
	case r.URL.Path == "/history/p1" && f.reads < 3:
		answer(w, map[string]any{})
	case r.URL.Path == "/history/p1" && f.failed:
		answer(w, map[string]any{"p1": map[string]any{"status": map[string]any{"status_str": "error", "completed": false,
			"messages": []any{[]any{"execution_error", map[string]any{"node_id": "598", "exception_message": "out of memory"}}}}, "outputs": map[string]any{}}})
	case r.URL.Path == "/history/p1":
		answer(w, map[string]any{"p1": map[string]any{
			"status": map[string]any{"status_str": "success", "completed": true, "messages": []any{
				[]any{"execution_start", map[string]any{"timestamp": 1000}}, []any{"execution_cached", map[string]any{"nodes": []any{"707:706"}, "timestamp": 1001}},
				[]any{"execution_success", map[string]any{"timestamp": 3500}},
			}},
			"outputs": map[string]map[string]any{
				"449":                   {"images": []any{map[string]any{"filename": "original.png", "subfolder": "MOSS/text_to_image", "type": "output"}}},
				latentNode:              {"latents": []any{map[string]any{"filename": "latent.latent", "subfolder": "overgo-recorder", "type": "output"}}},
				"35":                    {"text": []any{"the final prompt"}},
				"overgo-sigma-text-598": {"text": []any{"[\n    1.0,\n    0.6123046875,\n    0.0\n]"}},
				"707:703:431:flag":      {"animated": []any{false}},
				"707:702:434": {"dims": []any{1024, 576}, "input_bboxes": []any{map[string]any{"x": 102, "y": 115, "width": 358, "height": 230,
					"metadata": map[string]any{"type": "obj", "desc": "a café table"}}}},
			},
		}})
	case r.URL.Path == "/view" && served[r.URL.Query().Get("filename")] != "":
		_, _ = w.Write([]byte(served[r.URL.Query().Get("filename")]))
	default:
		http.NotFound(w, r)
	}
}

// socket is ComfyUI's websocket for the recorder: an event of no prompt at once, and once the prompt is posted, its
// start, its sampler's first step, a preview (binary, longer than a one-byte length), a ping, its second step, a step of
// another prompt and its success; then it reads the recorder's frames until the recorder closes.
func (f *fakeComfyUI) socket(w http.ResponseWriter, r *http.Request) {
	if r.URL.Query().Get("clientId") != clientID {
		http.Error(w, "unknown client", http.StatusForbidden)
		return
	}
	accept := sha1.Sum([]byte(r.Header.Get("Sec-WebSocket-Key") + socketAccept))
	conn, buffered, err := http.NewResponseController(w).Hijack()
	if err != nil {
		return
	}
	defer conn.Close()
	fmt.Fprintf(buffered, "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: %s\r\n\r\n", base64.StdEncoding.EncodeToString(accept[:]))
	frame := func(opcode byte, payload []byte) {
		header := []byte{0x80 | opcode}
		if len(payload) < 126 {
			header = append(header, byte(len(payload)))
		} else {
			header = binary.BigEndian.AppendUint16(append(header, 126), uint16(len(payload)))
		}
		_, _ = buffered.Write(append(header, payload...))
	}
	event := func(kind string, data map[string]any) {
		encoded, _ := json.Marshal(map[string]any{"type": kind, "data": data})
		frame(opText, encoded)
	}
	event("status", map[string]any{"status": map[string]any{"exec_info": map[string]any{"queue_remaining": 0}}})
	_ = buffered.Flush()
	select {
	case <-f.accepted:
	case <-f.quit:
		return
	}
	event("execution_start", map[string]any{"prompt_id": "p1", "timestamp": 1000})
	event("progress", map[string]any{"value": 1, "max": 2, "prompt_id": "p1", "node": "598"})
	frame(opBinary, make([]byte, 300))
	frame(opPing, []byte("still there"))
	event("progress", map[string]any{"value": 2, "max": 2, "prompt_id": "p1", "node": "598"})
	event("progress", map[string]any{"value": 1, "max": 1, "prompt_id": "another", "node": "598"})
	event("execution_success", map[string]any{"prompt_id": "p1", "timestamp": 3500})
	_ = buffered.Flush()
	// The recorder's first frame is its pong, masked.
	var header [2]byte
	if _, err := io.ReadFull(buffered, header[:]); err == nil && header[0]&0x0F == opPong && header[1]&0x80 != 0 {
		var mask [4]byte
		payload := make([]byte, header[1]&0x7F)
		if _, err := io.ReadFull(buffered, mask[:]); err == nil {
			if _, err := io.ReadFull(buffered, payload); err == nil {
				for index := range payload {
					payload[index] ^= mask[index%4]
				}
				f.mu.Lock()
				f.ponged = payload
				f.mu.Unlock()
			}
		}
	}
	_, _ = io.Copy(io.Discard, bufio.NewReader(buffered))
}

// TestARecordingIsTakenFromComfyUIsAnswers holds the recorder's host part to ComfyUI's HTTP API and websocket
// (comfy-recorder/client, comfy-recorder/record-rest), run as the command with its flags parsed afresh: it posts ComfyUI's
// own request for the section with the inputs given over it, a latent saver after the node named and the sigma nodes
// beside the sampler, waits on the history, reads the card's memory while it waits, and keeps each file the run wrote
// by its SHA-256 (an output of objects that name no file, as Create Bounding Boxes' boxes, fetching nothing), each text, the sampler's sigmas, each step as the websocket told it (a preview passed over, a ping
// answered, another prompt's step left out), each model file by the SHA-256 the recipe pins, the run's time as
// ComfyUI stamps it, and the commit of the install and of each custom-node pack it holds as a checkout. A run that
// fails in ComfyUI is refused with what ComfyUI said. A recording interrupted after ComfyUI accepted its prompt resumes
// that prompt, posting nothing; and a folder that holds the recording is refused before anything is posted. It is not
// parallel: the flags and arguments it sets are the process's.
func TestARecordingIsTakenFromComfyUIsAnswers(t *testing.T) {
	root := testutil.RepoRoot(t)
	install := t.TempDir()
	commit := func(folder string) string {
		testcheck.NoError(t, os.MkdirAll(folder, 0o755))
		for _, arguments := range [][]string{{"init", "-q"}, {"-c", "user.name=fixture", "-c", "user.email=fixture@example.com", "commit", "-q", "--allow-empty", "-m", "fixture"}} {
			testcheck.NoError(t, exec.CommandContext(t.Context(), "git", append([]string{"-C", folder}, arguments...)...).Run())
		}
		head := testcheck.Must(exec.CommandContext(t.Context(), "git", "-C", folder, "rev-parse", "HEAD").Output())(t)
		return strings.TrimSpace(string(head))
	}
	want := map[string]string{".": commit(install), "custom_nodes/pack": commit(filepath.Join(install, "custom_nodes", "pack"))}
	testcheck.NoError(t, os.MkdirAll(filepath.Join(install, "custom_nodes", "loose"), 0o755))
	recipePath := filepath.Join(root, "internal", "modelrecipe", "moss_ultra_import.json")
	recipe := testcheck.Must(modelartifact.ReadLocalImport(recipePath))(t)
	pins := map[string]string{}
	for _, file := range recipe.Files {
		pins[filepath.Base(file.Path)] = file.SHA256
	}

	record := func(t *testing.T, comfy *fakeComfyUI, out string) error {
		t.Helper()
		server := httptest.NewServer(comfy)
		t.Cleanup(func() { close(comfy.quit) })
		t.Cleanup(server.Close)
		saved, arguments := flag.CommandLine, os.Args
		t.Cleanup(func() { flag.CommandLine, os.Args = saved, arguments })
		flag.CommandLine = flag.NewFlagSet("comfy-reference", flag.ContinueOnError)
		os.Args = []string{"comfy-reference", "-fixture", filepath.Join(root, ultraFixture), "-record", "TEXT TO IMAGE", "-server", server.URL,
			"-install", install, "-out", out, "-recipe", recipePath, "-set", "598.seed=7", "-save-latent", "598"}
		return run()
	}
	read := func(t *testing.T, out string) recorded {
		t.Helper()
		var taken recorded
		testcheck.NoError(t, strictjson.DecodeBytes(testcheck.Must(os.ReadFile(filepath.Join(out, "text-to-image.json")))(t), &taken))
		return taken
	}

	t.Run("a run that succeeds", func(t *testing.T) {
		comfy, out := newFake(900, 700, 300, 800), t.TempDir()
		if err := record(t, comfy, out); err != nil {
			t.Fatalf("the recording failed: %v", err)
		}
		if seed := comfy.posted["598"].Inputs["seed"]; seed != 7.0 {
			t.Errorf("the request posted seed %v at the sampler; want the 7 given", seed)
		}
		if saver := comfy.posted[latentNode]; saver.ClassType != "SaveLatent" || !slices.Equal(saver.Inputs["samples"].([]any), []any{"598", 0.0}) {
			t.Errorf("the request posted %+v as the latent saver; want SaveLatent fed by the sampler's first output", saver)
		}
		sampled, scheduler := comfy.posted["598"], comfy.posted["overgo-sigmas-598"]
		for _, input := range []string{"model", "scheduler", "steps", "denoise"} {
			if fmt.Sprint(scheduler.Inputs[input]) != fmt.Sprint(sampled.Inputs[input]) {
				t.Errorf("the sigma scheduler is given %s %v; want the sampler's own, %v", input, scheduler.Inputs[input], sampled.Inputs[input])
			}
		}
		if scheduler.ClassType != "BasicScheduler" || comfy.posted["overgo-sigma-floats-598"].ClassType != "SigmasToFloat" || comfy.posted["overgo-sigma-text-598"].ClassType != "PreviewAny" {
			t.Errorf("the request posted the sigma nodes %+v, %+v, %+v; want BasicScheduler, SigmasToFloat and PreviewAny",
				scheduler, comfy.posted["overgo-sigma-floats-598"], comfy.posted["overgo-sigma-text-598"])
		}
		taken := read(t, out)
		var kept source
		testcheck.NoError(t, readJSON(filepath.Join(root, ultraFixture, sourceFile), &kept))
		if taken.Workflow != kept.SHA256 || taken.Section != "TEXT TO IMAGE" || taken.Prompt != "p1" || taken.ComfyUI != "0.3.fake" || taken.Resumed {
			t.Errorf("the recording names workflow %s, section %q, prompt %q, ComfyUI %q, resumed %t; want %s, TEXT TO IMAGE, p1, 0.3.fake, not resumed",
				taken.Workflow, taken.Section, taken.Prompt, taken.ComfyUI, taken.Resumed, kept.SHA256)
		}
		if taken.Milliseconds != 2500 || taken.PeakCardBytes != 700 {
			t.Errorf("the recording says %d ms and %d bytes of the card at most; want 2500, as ComfyUI stamped the run, and 700, 1000 less the least free", taken.Milliseconds, taken.PeakCardBytes)
		}
		if len(taken.Commits) != len(want) || taken.Commits["."] != want["."] || taken.Commits["custom_nodes/pack"] != want["custom_nodes/pack"] {
			t.Errorf("the recording names the commits %v; want %v, a pack that is no checkout left out", taken.Commits, want)
		}
		if !slices.Equal(taken.Cached, []string{"707:706"}) {
			t.Errorf("the recording names %v as answered from ComfyUI's cache; want 707:706, as its history said", taken.Cached)
		}
		if !slices.Equal(taken.Texts["35"], []string{"the final prompt"}) || taken.Set["598.seed"] != 7.0 || len(taken.Texts) != 1 {
			t.Errorf("the recording keeps the texts %v and the inputs set %v; want node 35's final prompt alone and 598.seed 7", taken.Texts, taken.Set)
		}
		if len(taken.Sigmas) != 1 || !slices.Equal(taken.Sigmas["598"], []float64{1, 0.6123046875, 0}) {
			t.Errorf("the recording keeps the sigmas %v; want the sampler 598's three, as its preview gave them", taken.Sigmas)
		}
		steps := taken.Progress["598"]
		if len(taken.Progress) != 1 || len(steps) != 2 || steps[0].Value != 1 || steps[1].Value != 2 || steps[1].Max != 2 || steps[0].Microseconds < 0 || steps[1].Microseconds < steps[0].Microseconds {
			t.Errorf("the recording keeps the progress %+v; want the sampler 598's two steps of p1 in order, another prompt's left out", taken.Progress)
		}
		if string(comfy.ponged) != "still there" {
			t.Errorf("the recorder answered the websocket's ping with %q; want its payload", comfy.ponged)
		}
		if len(taken.Models) != 3 {
			t.Errorf("the recording names the models %+v; want the section's diffusion model, encoder and VAE", taken.Models)
		}
		for _, model := range taken.Models {
			if model.SHA256 != pins[model.Name] || !slices.Contains(listed[model.Folder], model.Name) {
				t.Errorf("the recording names %+v; want the SHA-256 the recipe pins, %s, in a folder ComfyUI lists it in", model, pins[model.Name])
			}
		}
		if len(taken.Files) != len(served) {
			t.Fatalf("the recording keeps %+v; want the picture and the latent", taken.Files)
		}
		for _, file := range taken.Files {
			data := testcheck.Must(os.ReadFile(filepath.Join(out, "text-to-image", file.Name)))(t)
			if file.SHA256 != fmt.Sprintf("%x", sha256.Sum256(data)) || file.Type != "output" || !slices.Contains([]string{served["original.png"], served["latent.latent"]}, string(data)) {
				t.Errorf("the recording keeps %+v holding %q; want a file the run wrote, by its SHA-256", file, data)
			}
		}
		if _, err := os.Stat(filepath.Join(out, "text-to-image.accepted.json")); !os.IsNotExist(err) {
			t.Errorf("the recording leaves its accepted prompt kept (%v); want it gone once the recording is written", err)
		}
		again := newFake(900)
		if err := record(t, again, out); err == nil || !strings.Contains(err.Error(), "already holds a recording") || again.posts != 0 {
			t.Errorf("a recording into a folder that holds one ended as %v after %d posts; want it refused before anything is posted", err, again.posts)
		}
	})
	t.Run("a run that fails in ComfyUI", func(t *testing.T) {
		comfy := newFake(900)
		comfy.failed = true
		err := record(t, comfy, t.TempDir())
		if err == nil || !strings.Contains(err.Error(), "ComfyUI's run failed") || !strings.Contains(err.Error(), "out of memory") {
			t.Fatalf("a run ComfyUI failed gave %v; want it refused with what ComfyUI said", err)
		}
	})
	t.Run("a recording interrupted after its prompt was accepted", func(t *testing.T) {
		out, cut := t.TempDir(), newFake(900)
		cut.interrupted = true
		if err := record(t, cut, out); err == nil || cut.posts != 1 {
			t.Fatalf("a recording whose ComfyUI stopped answering ended as %v after %d posts; want it to fail after posting once", err, cut.posts)
		}
		resumed := newFake(900, 700, 300, 800)
		if err := record(t, resumed, out); err != nil || resumed.posts != 0 {
			t.Fatalf("the recording again ended as %v after %d posts; want the accepted prompt resumed, nothing posted", err, resumed.posts)
		}
		if taken := read(t, out); taken.Prompt != "p1" || !taken.Resumed || len(taken.Progress) != 0 || len(taken.Sigmas["598"]) != 3 {
			t.Errorf("the resumed recording names prompt %q, resumed %t, progress %v, sigmas %v; want p1 resumed, no steps it did not hear, and its sigmas",
				taken.Prompt, taken.Resumed, taken.Progress, taken.Sigmas)
		}
	})
}

// TestTheModelsARequestLoadsAreNamedByTheirPins holds the recorder's naming of model files (comfy-recorder/record-rest):
// each file of the recipe a request names, at a node's input or nested in one, is named with the SHA-256 the recipe
// pins and the folder its source path names; an adapter switched off is not; and a file ComfyUI does not list in that
// folder is refused.
func TestTheModelsARequestLoadsAreNamedByTheirPins(t *testing.T) {
	recipe := testcheck.Must(modelartifact.ReadLocalImport(filepath.Join(testutil.RepoRoot(t), "internal", "modelrecipe", "moss_ultra_import.json")))(t)
	server := httptest.NewServer(newFake(900))
	t.Cleanup(server.Close)
	r := recorder{server: server.URL}
	request := map[string]apiNode{
		"1": {ClassType: "UNETLoader", Inputs: map[string]any{"unet_name": "moss_image_base_int8_convrot.safetensors", "weight_dtype": "default"}},
		"2": {ClassType: "Power Lora Loader (rgthree)", Inputs: map[string]any{
			"lora_1": map[string]any{"on": false, "lora": "8-steps-dmd2-lora.safetensors", "strength": 1.0},
			"lora_2": map[string]any{"on": true, "lora": "moss_image_sft.safetensors", "strength": 1.0},
		}},
	}
	named := testcheck.Must(r.models(t.Context(), recipe, request))(t)
	wanted := []recordedModel{{Node: "1", Input: "unet_name", Folder: "diffusion_models", Name: "moss_image_base_int8_convrot.safetensors"}, {Node: "2", Input: "lora_2.lora", Folder: "loras", Name: "moss_image_sft.safetensors"}}
	for index := range wanted {
		for _, file := range recipe.Files {
			if filepath.Base(file.Path) == wanted[index].Name {
				wanted[index].SHA256 = file.SHA256
			}
		}
	}
	if !slices.Equal(named, wanted) {
		t.Errorf("the request names the models %+v; want %+v, the adapter switched off left out", named, wanted)
	}
	request["2"].Inputs["lora_1"].(map[string]any)["on"] = true
	if _, err := r.models(t.Context(), recipe, request); err == nil || !strings.Contains(err.Error(), "does not list") {
		t.Errorf("a request naming a file ComfyUI does not list ended as %v; want it refused", err)
	}
}

// TestAMalformedOrMissingRequestIsRefused holds the recorder to ComfyUI's own requests (SQA-345): a section's request
// is the file ComfyUI's page exported, read strictly, and one that does not parse, holds no node, holds a node with no
// class, carries a field ComfyUI does not export, or is not there is refused before anything runs.
func TestAMalformedOrMissingRequestIsRefused(t *testing.T) {
	t.Parallel()
	for name, data := range map[string]string{
		"not JSON":         `{"1": `,
		"no node":          `{}`,
		"a node no class":  `{"1": {"inputs": {}}}`,
		"an unknown field": `{"1": {"class_type": "KSampler", "inputs": {}, "widgets": []}}`,
	} {
		if _, err := parseRequest([]byte(data)); err == nil {
			t.Errorf("a request with %s was read; want it refused", name)
		}
	}
	folder := t.TempDir()
	missing := &fixture{folder: folder, sections: map[string]section{"TEXT TO IMAGE": {}}, requests: map[string]string{"TEXT TO IMAGE": "requests/text-to-image.json"}}
	if _, err := missing.request("TEXT TO IMAGE"); err == nil || !errors.Is(err, os.ErrNotExist) {
		t.Errorf("a section whose request is not there ended as %v; want its absence", err)
	}
	if _, err := missing.request("DEPTH"); err == nil || !strings.Contains(err.Error(), "has no section") {
		t.Errorf("a section the fixture does not hold ended as %v; want it refused", err)
	}
}
