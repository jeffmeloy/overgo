package main

import (
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"overgo/internal/strictjson"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// fakeComfyUI answers as ComfyUI's HTTP API does for one prompt: its stats with the card's free memory falling and
// rising, the queue taking the prompt, a history with no entry and then the run's, and each file the run wrote. It
// keeps the request posted.
type fakeComfyUI struct {
	failed bool
	mu     sync.Mutex
	posted map[string]apiNode
	reads  int
	free   []uint64
}

// answer writes value as a JSON answer.
func answer(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}

// served are the bytes of the files the fake's run wrote, by name.
var served = map[string]string{"original.png": "picture bytes", "latent.latent": "latent bytes"}

// latentNode is the node a recording adds to save the sampler's latent, named as the command names it.
const latentNode = "overgo-latent-598"

// recorded is a recording as the command writes it, read here by its own fields, so the test builds where the command
// writes none.
type recorded struct {
	Workflow      string              `json:"workflow_sha256"`
	Section       string              `json:"section"`
	Request       string              `json:"request_sha256"`
	Set           map[string]any      `json:"set"`
	ComfyUI       string              `json:"comfyui_version"`
	Commits       map[string]string   `json:"commits"`
	Prompt        string              `json:"prompt_id"`
	Milliseconds  int64               `json:"milliseconds"`
	PeakCardBytes uint64              `json:"peak_card_bytes"`
	Texts         map[string][]string `json:"texts"`
	Files         []struct {
		Node, Kind, Name, SHA256 string
		Bytes                    int
	} `json:"files"`
}

func (f *fakeComfyUI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	switch {
	case r.URL.Path == "/system_stats":
		free := f.free[min(f.reads, len(f.free)-1)]
		f.reads++
		answer(w, map[string]any{"system": map[string]any{"comfyui_version": "0.3.fake"}, "devices": []any{map[string]any{"vram_total": 1000, "vram_free": free}}})
	case r.URL.Path == "/prompt":
		var body struct {
			Prompt map[string]apiNode `json:"prompt"`
		}
		if json.NewDecoder(r.Body).Decode(&body) != nil {
			http.Error(w, "unreadable", http.StatusBadRequest)
			return
		}
		f.posted = body.Prompt
		answer(w, map[string]any{"prompt_id": "p1", "number": 0, "node_errors": map[string]any{}})
	case r.URL.Path == "/history/p1" && f.reads < 3:
		answer(w, map[string]any{})
	case r.URL.Path == "/history/p1" && f.failed:
		answer(w, map[string]any{"p1": map[string]any{"status": map[string]any{"status_str": "error", "completed": false,
			"messages": []any{[]any{"execution_error", map[string]any{"node_id": "598", "exception_message": "out of memory"}}}}, "outputs": map[string]any{}}})
	case r.URL.Path == "/history/p1":
		answer(w, map[string]any{"p1": map[string]any{
			"status": map[string]any{"status_str": "success", "completed": true, "messages": []any{
				[]any{"execution_start", map[string]any{"timestamp": 1000}}, []any{"execution_cached", map[string]any{"nodes": []any{}}},
				[]any{"execution_success", map[string]any{"timestamp": 3500}},
			}},
			"outputs": map[string]map[string]any{
				"449":              {"images": []any{map[string]any{"filename": "original.png", "subfolder": "MOSS/text_to_image", "type": "output"}}},
				latentNode:         {"latents": []any{map[string]any{"filename": "latent.latent", "subfolder": "overgo-recorder", "type": "output"}}},
				"35":               {"text": []any{"the final prompt"}},
				"707:703:431:flag": {"animated": []any{false}},
			},
		}})
	case r.URL.Path == "/view" && served[r.URL.Query().Get("filename")] != "":
		_, _ = w.Write([]byte(served[r.URL.Query().Get("filename")]))
	default:
		http.NotFound(w, r)
	}
}

// TestARecordingIsTakenFromComfyUIsAnswers holds the recorder's host part to ComfyUI's HTTP API
// (comfy-recorder/client), run as the command with its flags parsed afresh: it posts the section's request with the
// inputs given over it and a latent saver after the node named, waits on the history, reads the card's memory while
// it waits, and keeps each file the run wrote by its SHA-256, each text, the run's time as ComfyUI stamps it, and the
// commit of the install and of each custom-node pack it holds as a checkout. A run that fails in ComfyUI is refused
// with what ComfyUI said. It is not parallel: the flags and arguments it sets are the process's.
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

	record := func(t *testing.T, comfy *fakeComfyUI) (string, error) {
		t.Helper()
		server := httptest.NewServer(comfy)
		t.Cleanup(server.Close)
		out := t.TempDir()
		saved, arguments := flag.CommandLine, os.Args
		t.Cleanup(func() { flag.CommandLine, os.Args = saved, arguments })
		flag.CommandLine = flag.NewFlagSet("comfy-reference", flag.ContinueOnError)
		os.Args = []string{"comfy-reference", "-fixture", filepath.Join(root, ultraFixture), "-record", "TEXT TO IMAGE", "-server", server.URL,
			"-install", install, "-out", out, "-set", "598.seed=7", "-save-latent", "598"}
		return out, run()
	}

	t.Run("a run that succeeds", func(t *testing.T) {
		comfy := &fakeComfyUI{free: []uint64{900, 700, 300, 800}}
		out, err := record(t, comfy)
		if err != nil {
			t.Fatalf("the recording failed: %v", err)
		}
		if seed := comfy.posted["598"].Inputs["seed"]; seed != 7.0 {
			t.Errorf("the request posted seed %v at the sampler; want the 7 given", seed)
		}
		if saver := comfy.posted[latentNode]; saver.ClassType != "SaveLatent" || !slices.Equal(saver.Inputs["samples"].([]any), []any{"598", 0.0}) {
			t.Errorf("the request posted %+v as the latent saver; want SaveLatent fed by the sampler's first output", saver)
		}
		var taken recorded
		testcheck.NoError(t, strictjson.DecodeBytes(testcheck.Must(os.ReadFile(filepath.Join(out, "text-to-image.json")))(t), &taken))
		var kept source
		testcheck.NoError(t, readJSON(filepath.Join(root, ultraFixture, sourceFile), &kept))
		if taken.Workflow != kept.SHA256 || taken.Section != "TEXT TO IMAGE" || taken.Prompt != "p1" || taken.ComfyUI != "0.3.fake" {
			t.Errorf("the recording names workflow %s, section %q, prompt %q, ComfyUI %q; want %s, TEXT TO IMAGE, p1, 0.3.fake", taken.Workflow, taken.Section, taken.Prompt, taken.ComfyUI, kept.SHA256)
		}
		if taken.Milliseconds != 2500 || taken.PeakCardBytes != 700 {
			t.Errorf("the recording says %d ms and %d bytes of the card at most; want 2500, as ComfyUI stamped the run, and 700, 1000 less the least free", taken.Milliseconds, taken.PeakCardBytes)
		}
		if len(taken.Commits) != len(want) || taken.Commits["."] != want["."] || taken.Commits["custom_nodes/pack"] != want["custom_nodes/pack"] {
			t.Errorf("the recording names the commits %v; want %v, a pack that is no checkout left out", taken.Commits, want)
		}
		if !slices.Equal(taken.Texts["35"], []string{"the final prompt"}) || taken.Set["598.seed"] != 7.0 {
			t.Errorf("the recording keeps the texts %v and the inputs set %v; want node 35's final prompt and 598.seed 7", taken.Texts, taken.Set)
		}
		if len(taken.Files) != len(served) {
			t.Fatalf("the recording keeps %+v; want the picture and the latent", taken.Files)
		}
		for _, file := range taken.Files {
			data := testcheck.Must(os.ReadFile(filepath.Join(out, "text-to-image", file.Name)))(t)
			if file.SHA256 != fmt.Sprintf("%x", sha256.Sum256(data)) || !slices.Contains([]string{served["original.png"], served["latent.latent"]}, string(data)) {
				t.Errorf("the recording keeps %+v holding %q; want a file the run wrote, by its SHA-256", file, data)
			}
		}
	})
	t.Run("a run that fails in ComfyUI", func(t *testing.T) {
		_, err := record(t, &fakeComfyUI{failed: true, free: []uint64{900}})
		if err == nil || !strings.Contains(err.Error(), "ComfyUI's run failed") || !strings.Contains(err.Error(), "out of memory") {
			t.Fatalf("a run ComfyUI failed gave %v; want it refused with what ComfyUI said", err)
		}
	})
}
