package webuilane

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"overgo/internal/testevidence"
	"path/filepath"
	"strings"
	"testing"
)

func TestWebUIBrowserDownloadIsolation(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testevidence.ShortIntegrationSkip + ": download isolation runs through cmd/webui-lane")
	}
	executable, err := FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	const payload = "isolated browser download"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/download" {
			w.Header().Set("Content-Disposition", `attachment; filename="fixture.txt"`)
			w.Write([]byte(payload))
			return
		}
		w.Header().Set("Content-Type", "text/html")
		w.Write([]byte(`<a id="download" href="/download" download="fixture.txt">Download</a>`))
	}))
	defer server.Close()
	browser, err := Open(t.Context(), executable, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	// Observe browser-level completion on a separate socket: page method calls
	// consume their own events while waiting for command responses.
	endpoint, err := os.ReadFile(filepath.Join(browser.profile, "DevToolsActivePort"))
	if err != nil {
		t.Fatal(err)
	}
	port, target, found := strings.Cut(strings.TrimSpace(string(endpoint)), "\n")
	if !found {
		t.Fatal("browser event endpoint is absent")
	}
	ctx := t.Context()
	if deadline, ok := t.Deadline(); ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithDeadlineCause(ctx, deadline, context.DeadlineExceeded)
		defer cancel()
	}
	events, err := dialWebSocket(ctx, "ws://127.0.0.1:"+strings.TrimSpace(port)+strings.TrimSpace(target))
	if err != nil {
		t.Fatal(err)
	}
	defer events.close()
	if err := events.call(ctx, "Browser.setDownloadBehavior", map[string]any{"behavior": "allow", "downloadPath": filepath.Join(browser.profile, "downloads"), "eventsEnabled": true}, nil); err != nil {
		t.Fatal(err)
	}
	if err := browser.Eventually(t.Context(), `!!document.querySelector('#download')`); err != nil {
		t.Fatal(err)
	}
	if err := browser.Evaluate(t.Context(), `(async()=>{const b=await (await fetch("/download")).blob();const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="fixture.txt";a.click()})()`, nil); err != nil {
		t.Fatal(err)
	}
	destination := filepath.Join(browser.profile, "downloads", "fixture.txt")
	guid := ""
	for {
		data, err := events.readText()
		if err != nil {
			t.Fatal(err)
		}
		var event struct {
			Method string `json:"method"`
			Params struct {
				GUID              string `json:"guid"`
				SuggestedFilename string `json:"suggestedFilename"`
				State             string `json:"state"`
			} `json:"params"`
		}
		if err := json.Unmarshal(data, &event); err != nil {
			t.Fatal(err)
		}
		if event.Method == "Browser.downloadWillBegin" && event.Params.SuggestedFilename == "fixture.txt" {
			guid = event.Params.GUID
		}
		if event.Method != "Browser.downloadProgress" || guid == "" || event.Params.GUID != guid {
			continue
		}
		if event.Params.State == "canceled" {
			t.Fatal("browser canceled the download")
		}
		if event.Params.State == "completed" {
			break
		}
	}
	data, err := os.ReadFile(destination)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != payload {
		t.Fatal("download bytes differ")
	}
	profile := browser.profile
	if err := browser.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(profile); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("download profile survives close: %v", err)
	}
}
