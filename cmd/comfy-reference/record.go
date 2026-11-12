package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/url"
	"os"

	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
)

// recording is what one run of a section in ComfyUI's own process gave: what ran (the workflow's hash, the section,
// the hash of the request posted and each input set over the section's own, ComfyUI's version and the commit of it
// and of each custom-node pack), how long ComfyUI says it ran, the most of the card's memory in use while it ran, and
// each output: a file by its node, kind, name and SHA-256, or a node's text.
type recording struct {
	Workflow      string              `json:"workflow_sha256"`
	Section       string              `json:"section"`
	Request       string              `json:"request_sha256"`
	Set           map[string]any      `json:"set,omitzero"`
	ComfyUI       string              `json:"comfyui_version"`
	Commits       map[string]string   `json:"commits"`
	Prompt        string              `json:"prompt_id"`
	Milliseconds  int64               `json:"milliseconds"`
	PeakCardBytes uint64              `json:"peak_card_bytes"`
	Files         []recordedFile      `json:"files"`
	Texts         map[string][]string `json:"texts,omitzero"`
}

// recordedFile is one file a run wrote, as ComfyUI's history names it, kept beside the recording.
type recordedFile struct {
	Node   string `json:"node"`
	Kind   string `json:"kind"`
	Name   string `json:"name"`
	SHA256 string `json:"sha256"`
	Bytes  int    `json:"bytes"`
}

// recorder runs a fixture's sections in a ComfyUI listening at server, an install's ComfyUI folder at install, and
// keeps each run's files and recording in out.
type recorder struct {
	ultra   *fixture
	server  string
	install string
	out     string
	// set are inputs given over the section's own, "node.input=value", the value JSON; latents are the nodes whose
	// first output, a latent, is saved beside the images.
	set, latents []string
	// poll is how often ComfyUI's history and memory are read while the run lasts.
	poll time.Duration
}

// latentSaver is the id of the node a recording adds to save node's latent.
func latentSaver(node string) string { return "overgo-latent-" + node }

// record runs the section titled title and returns its recording, written to out as <section>.json beside its files.
func (r recorder) record(ctx context.Context, title string) (recording, error) {
	request, err := r.ultra.request(title)
	if err != nil {
		return recording{}, err
	}
	taken := recording{Workflow: hashOf(r.ultra.workflow), Section: title, Set: map[string]any{}, Commits: map[string]string{}, Texts: map[string][]string{}}
	for _, setting := range r.set {
		at, value, given := strings.Cut(setting, "=")
		node, input, named := strings.Cut(at, ".")
		var decoded any
		if !given || !named || json.Unmarshal([]byte(value), &decoded) != nil {
			return recording{}, fmt.Errorf("comfy-reference: -set %q is not node.input=<JSON value>", setting)
		}
		if _, held := request[node].Inputs[input]; !held {
			return recording{}, fmt.Errorf("comfy-reference: -set %q names no input of the section's request", setting)
		}
		request[node].Inputs[input], taken.Set[at] = decoded, decoded
	}
	for _, node := range r.latents {
		if _, held := request[node]; !held {
			return recording{}, fmt.Errorf("comfy-reference: -save-latent %q names no node of the section's request", node)
		}
		request[latentSaver(node)] = apiNode{ClassType: "SaveLatent", Inputs: map[string]any{"samples": []any{node, 0}, "filename_prefix": "overgo-recorder/" + slug(title)}}
	}
	posted, err := json.Marshal(request)
	if err != nil {
		return recording{}, err
	}
	taken.Request = hashOf(posted)
	if taken.Commits, err = commits(ctx, r.install); err != nil {
		return recording{}, err
	}
	stats, err := r.stats(ctx)
	if err != nil {
		return recording{}, err
	}
	taken.ComfyUI = stats.System.Version
	var queued struct {
		Prompt string                     `json:"prompt_id"`
		Errors map[string]json.RawMessage `json:"node_errors"`
	}
	if err := r.call(ctx, http.MethodPost, "/prompt", map[string]any{"prompt": json.RawMessage(posted), "client_id": "overgo-comfy-reference"}, &queued); err != nil {
		return recording{}, err
	}
	if len(queued.Errors) > 0 || queued.Prompt == "" {
		return recording{}, fmt.Errorf("comfy-reference: ComfyUI refused the request of %q: %v", title, queued.Errors)
	}
	taken.Prompt = queued.Prompt
	run, peak, err := r.await(ctx, queued.Prompt, stats)
	if err != nil {
		return recording{}, fmt.Errorf("comfy-reference: %q: %w", title, err)
	}
	taken.PeakCardBytes = peak
	taken.Milliseconds, err = run.elapsed()
	if err != nil {
		return recording{}, err
	}
	folder := filepath.Join(r.out, slug(title))
	if err := os.MkdirAll(folder, 0o755); err != nil {
		return recording{}, err
	}
	for _, node := range slices.Sorted(maps.Keys(run.Outputs)) {
		for _, kind := range slices.Sorted(maps.Keys(run.Outputs[node])) {
			raw := run.Outputs[node][kind]
			var texts []string
			if json.Unmarshal(raw, &texts) == nil {
				taken.Texts[node] = append(taken.Texts[node], texts...)
				continue
			}
			var files []struct{ Filename, Subfolder, Type string }
			if json.Unmarshal(raw, &files) != nil {
				continue // an output that is neither files nor text (a flag, a count) names nothing to keep
			}
			for _, file := range files {
				kept, err := r.keep(ctx, folder, node, kind, file.Filename, file.Subfolder, file.Type)
				if err != nil {
					return recording{}, err
				}
				taken.Files = append(taken.Files, kept)
			}
		}
	}
	return taken, jsonfile.Write(filepath.Join(r.out, slug(title)+".json"), taken, fixtureMode)
}

// history is one prompt's entry in ComfyUI's history: whether it ended, how, the events it passed with their times,
// and each output node's outputs by kind.
type history struct {
	Status struct {
		Status    string              `json:"status_str"`
		Completed bool                `json:"completed"`
		Messages  [][]json.RawMessage `json:"messages"`
	} `json:"status"`
	Outputs map[string]map[string]json.RawMessage `json:"outputs"`
}

// elapsed is the time from the run's start to its success, in milliseconds, as ComfyUI stamps the two events.
func (h history) elapsed() (int64, error) {
	stamps := map[string]int64{}
	for _, message := range h.Status.Messages {
		var event string
		var detail struct {
			Timestamp int64 `json:"timestamp"`
		}
		if len(message) == 2 && json.Unmarshal(message[0], &event) == nil && json.Unmarshal(message[1], &detail) == nil {
			stamps[event] = detail.Timestamp
		}
	}
	start, started := stamps["execution_start"]
	end, ended := stamps["execution_success"]
	if !started || !ended {
		return 0, errors.New("comfy-reference: ComfyUI's history stamps no start and success of the run")
	}
	return end - start, nil
}

// systemStats is what ComfyUI's /system_stats says of itself and the card.
type systemStats struct {
	System struct {
		Version string `json:"comfyui_version"`
	} `json:"system"`
	Devices []struct {
		Total uint64 `json:"vram_total"`
		Free  uint64 `json:"vram_free"`
	} `json:"devices"`
}

func (r recorder) stats(ctx context.Context) (systemStats, error) {
	var stats systemStats
	if err := r.call(ctx, http.MethodGet, "/system_stats", nil, &stats); err != nil {
		return systemStats{}, err
	}
	if len(stats.Devices) == 0 {
		return systemStats{}, errors.New("comfy-reference: ComfyUI's /system_stats names no card")
	}
	return stats, nil
}

// await reads prompt's history until it ends, and ComfyUI's memory as often, and returns the history and the most of
// the card's memory in use meanwhile, read by the free memory ComfyUI reports.
func (r recorder) await(ctx context.Context, prompt string, before systemStats) (history, uint64, error) {
	peak := before.Devices[0].Total - before.Devices[0].Free
	ticker := time.NewTicker(r.poll)
	defer ticker.Stop()
	for {
		var entries map[string]history
		if err := r.call(ctx, http.MethodGet, "/history/"+url.PathEscape(prompt), nil, &entries); err != nil {
			return history{}, 0, err
		}
		if run, held := entries[prompt]; held && (run.Status.Completed || run.Status.Status == "error") {
			if run.Status.Status == "error" {
				return history{}, 0, fmt.Errorf("ComfyUI's run failed: %s", run.Status.Messages)
			}
			return run, peak, nil
		}
		stats, err := r.stats(ctx)
		if err != nil {
			return history{}, 0, err
		}
		peak = max(peak, stats.Devices[0].Total-stats.Devices[0].Free)
		select {
		case <-ctx.Done():
			return history{}, 0, ctx.Err()
		case <-ticker.C:
		}
	}
}

// keep fetches one output file through ComfyUI's /view and writes it into folder, named by its node and file name.
func (r recorder) keep(ctx context.Context, folder, node, kind, name, subfolder, place string) (recordedFile, error) {
	query := url.Values{"filename": {name}, "subfolder": {subfolder}, "type": {place}}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, r.server+"/view?"+query.Encode(), nil)
	if err != nil {
		return recordedFile{}, err
	}
	data, err := fetch(request)
	if err != nil {
		return recordedFile{}, err
	}
	kept := recordedFile{Node: node, Kind: kind, Name: strings.NewReplacer(":", "-", "/", "-").Replace(node) + "-" + filepath.Base(name), SHA256: hashOf(data), Bytes: len(data)}
	return kept, os.WriteFile(filepath.Join(folder, kept.Name), data, fixtureMode)
}

// call sends body, JSON where given, to ComfyUI at path and decodes its JSON answer into answer.
func (r recorder) call(ctx context.Context, method, path string, body, answer any) error {
	var sent io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return err
		}
		sent = bytes.NewReader(encoded)
	}
	request, err := http.NewRequestWithContext(ctx, method, r.server+path, sent)
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	data, err := fetch(request)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, answer)
}

// fetch sends request and returns the answer's body, refused unless ComfyUI answered 200.
func fetch(request *http.Request) ([]byte, error) {
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	data, err := io.ReadAll(response.Body)
	if err == nil && response.StatusCode != http.StatusOK {
		err = fmt.Errorf("comfy-reference: ComfyUI answered %s %s with %s: %s", request.Method, request.URL.Path, response.Status, strings.TrimSpace(string(data)))
	}
	return data, err
}

// commits are the commit of the install's ComfyUI and of each custom-node pack it holds as a git checkout, by folder.
func commits(ctx context.Context, install string) (map[string]string, error) {
	folders := []string{"."}
	packs, err := os.ReadDir(filepath.Join(install, "custom_nodes"))
	if err != nil {
		return nil, err
	}
	for _, pack := range packs {
		if _, err := os.Stat(filepath.Join(install, "custom_nodes", pack.Name(), ".git")); err == nil {
			folders = append(folders, "custom_nodes/"+pack.Name())
		}
	}
	found := map[string]string{}
	for _, folder := range folders {
		output, err := gitauthority.Query(ctx, filepath.Join(install, filepath.FromSlash(folder)), "rev-parse", "HEAD")
		if err != nil {
			return nil, fmt.Errorf("comfy-reference: the commit of %s: %w", folder, err)
		}
		found[folder] = strings.TrimSpace(string(output))
	}
	return found, nil
}

// hashOf is data's SHA-256 in hex.
func hashOf(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
