package webuilane

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
	"path/filepath"
	"strings"
	"testing"
)

func TestWebUIBrowserDownloadIsolation(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		testskip.NotApplicable(t, "download isolation runs through cmd/webui-lane")
	}
	executable, err := FindBrowser(os.Getenv("OVERGO_BROWSER"))
	testcheck.NoError(t, err)
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
	testcheck.NoError(t, err)
	defer browser.Close()
	// Observe browser-level completion on a separate socket: page method calls
	// consume their own events while waiting for command responses.
	endpoint, err := os.ReadFile(filepath.Join(browser.profile, "DevToolsActivePort"))
	testcheck.NoError(t, err)
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
	testcheck.NoError(t, err)
	defer events.close()
	testcheck.NoError(t, events.call(ctx, "Browser.setDownloadBehavior", map[string]any{"behavior": "allow", "downloadPath": filepath.Join(browser.profile, "downloads"), "eventsEnabled": true}, nil))
	testcheck.NoError(t, browser.Eventually(t.Context(), `!!document.querySelector('#download')`))
	testcheck.NoError(t, browser.Evaluate(t.Context(), `(async()=>{const b=await (await fetch("/download")).blob();const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="fixture.txt";a.click()})()`, nil))
	destination := filepath.Join(browser.profile, "downloads", "fixture.txt")
	guid := ""
	for {
		data, err := events.readText()
		testcheck.NoError(t, err)
		var event struct {
			Method string `json:"method"`
			Params struct {
				GUID              string `json:"guid"`
				SuggestedFilename string `json:"suggestedFilename"`
				State             string `json:"state"`
			} `json:"params"`
		}
		testcheck.NoError(t, json.Unmarshal(data, &event))
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
	testcheck.NoError(t, err)
	if string(data) != payload {
		t.Fatal("download bytes differ")
	}
	profile := browser.profile
	testcheck.NoError(t, browser.Close())
	if _, err := os.Stat(profile); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("download profile survives close: %v", err)
	}
}
