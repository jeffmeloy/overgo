package server

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/libraryintake"
	"overgo/internal/modelswap"
	"overgo/internal/operation"
	"overgo/internal/overgodbtest"
	"overgo/internal/processcontrol"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

const laneHubRepository = "lane/validated"

// hubFixtureServer serves the lane repository whose single file is file, the
// shape the workbench's discovery surface consumes; a journey hands it a real
// GGUF, a page leg a few fixture bytes.
func hubFixtureServer(t *testing.T, file string) *httptest.Server {
	t.Helper()
	handle := testcheck.Must(os.Open(file))(t)
	defer handle.Close()
	hash := sha256.New()
	size := testcheck.Must(io.Copy(hash, handle))(t)
	name := filepath.Base(file)
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/models", func(response http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(response).Encode([]map[string]any{{"id": laneHubRepository, "downloads": 1}})
	})
	mux.HandleFunc("GET /api/models/lane/validated", func(response http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(response).Encode(map[string]any{
			"id": laneHubRepository, "sha": "rev0",
			"siblings": []map[string]any{{"rfilename": name, "lfs": map[string]any{"sha256": hex.EncodeToString(hash.Sum(nil)), "size": size}}},
		})
	})
	mux.HandleFunc("GET /lane/validated/resolve/rev0/{file}", func(response http.ResponseWriter, request *http.Request) {
		if request.PathValue("file") != name {
			http.NotFound(response, request)
			return
		}
		http.ServeFile(response, request, file)
	})
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	return server
}

// TestModelJourneyLibraryValidation: from a cold proxy over an empty store
// the Library downloads a real small GGUF from the hub, registers it,
// validates it as an operation the workbench behind the shell runs, and
// the picker then serves it; a model registered with its projector
// validates its text path and keeps the projector as its projection.
func TestModelJourneyLibraryValidation(t *testing.T) {
	t.Parallel()
	if os.Getenv(webuilane.ModelJourneyEnvironment) != "1" {
		testskip.NotApplicable(t, "library validation runs through cmd/webui-lane -journeys")
	}
	root := testutil.RepoRoot(t)
	roots := testcheck.Must(dataroot.Resolve(root))(t)
	small := roots.ResolveModelPath("overgo-hfconvert/Qwen2.5-0.5B-f16.gguf")
	if _, err := os.Stat(small); err != nil {
		t.Fatalf("library validation unavailable: the small text model is absent: %v", err)
	}
	ctx := t.Context()
	projectedName, projectedLocation, projectorPath := smallestDeclaredProjector(t, ctx, roots.Store)
	if projectedLocation == "" {
		t.Fatal("library validation unavailable: no model with a declared projector has bytes on disk")
	}
	binary := filepath.Join(t.TempDir(), "overgo-server.exe")
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "go", Args: []string{"build", "-o", binary, "./cmd/server"}, Dir: root, Stdout: os.Stderr, Stderr: os.Stderr,
	})
	if err != nil || receipt.ExitCode != 0 {
		t.Fatalf("library validation unavailable: the server binary did not build: %v (exit %d)", err, receipt.ExitCode)
	}
	hub := hubFixtureServer(t, small)
	// The private store starts as the data root's copy: the registered
	// architecture profiles the Library registers against live there, and
	// nothing the journey writes reaches the data root.
	storeDir := filepath.Join(t.TempDir(), "store")
	testcheck.NoError(t, prepareBrowserJourneyStore(ctx, roots.Store, storeDir))
	store := overgodbtest.Create(t, storeDir)
	intake := LibraryIntake{ModelFiles: libraryintake.ModelFiles, Register: libraryintake.Register, Validate: libraryintake.Validate}
	workbench := testcheck.Must(New(Config{
		RuntimePolicy: testRuntimePolicy(), Repository: store, LibraryIntake: intake,
		HubEndpoint: hub.URL, HubDownloadRoot: filepath.Join(t.TempDir(), "downloads"),
	}, GenerationRefused{Reason: idleRefusal}))(t)
	defer workbench.Close()
	testcheck.NoError(t, os.MkdirAll(workbench.config.HubDownloadRoot, 0o755))
	supervisor := testcheck.Must(modelswap.New(modelswap.ServerLauncher{Binary: binary, Store: storeDir, Dir: root}, 0))(t)
	defer supervisor.Close()
	catalog := overgodbtest.OpenReadOnly(t, storeDir)
	resolver := &modelswap.CatalogResolver{Store: catalog, Limit: 256}
	proxy := &modelswap.Proxy{
		Supervisor: supervisor, Resolver: resolver, Keys: resolver,
		Idle: &IdleShell{Catalog: resolver.Catalog, Repository: store, Intake: intake, Workbench: workbench},
	}
	front := httptest.NewServer(proxy)
	defer front.Close()
	path := testcheck.Must(webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER")))(t)
	browser := testcheck.Must(webuilane.Open(ctx, path, front.URL+"/"))(t)
	defer browser.Close()
	testcheck.NoError(t, browser.SetViewport(ctx, webuilane.ScreenViewports[0].Width, webuilane.ScreenViewports[0].Height))
	// API idleness does not imply shell readiness: workspace scripts load
	// outside the API request counter. Use the shared browser predicate wait.
	settle := func(what, expression string) {
		t.Helper()
		if err := browser.Eventually(ctx, expression); err != nil {
			var page string
			_ = browser.Evaluate(ctx, `JSON.stringify({errors: window.overgo && window.overgo.errors, banners: [...document.querySelectorAll('#panel-library .err-banner')].map((node) => node.textContent), inFlight: window.overgo.api.inFlight(), rows: [...document.querySelectorAll('#panel-library tr')].map((row) => row.outerHTML.slice(0, 400))})`, &page)
			t.Fatalf("%s: %v; page: %s", what, err, page)
		}
	}
	rowNote := func(rowText, text string) string {
		return `[...document.querySelectorAll("#panel-library tr")].some((row) => row.textContent.includes(` + strconv.Quote(rowText) + `) && row.textContent.includes(` + strconv.Quote(text) + `))`
	}
	// A row's request that fails stands in the row as the page's failure banner: a wait ends at it, and the leg fails
	// with its words, where a wait for the row's success alone outlived an HTTP 500.
	rowFailure := func(rowText string) string {
		return `[...document.querySelectorAll("#panel-library tr")].find((row) => row.textContent.includes(` + strconv.Quote(rowText) + `) && row.querySelector(".err-banner"))`
	}
	failIfRowFailed := func(rowText, what string) {
		t.Helper()
		if failure := pageText(t, ctx, browser, `(`+rowFailure(rowText)+`)?.querySelector(".err-banner")?.textContent || ""`); failure != "" {
			t.Fatalf("%s failed: %s", what, failure)
		}
	}
	rowAnswers := func(rowText, text, what string) {
		t.Helper()
		settle(what, rowNote(rowText, text)+` || !!`+rowFailure(rowText))
		failIfRowFailed(rowText, what)
	}
	// A validation runs to completion, or the leg fails with what the row answered.
	validated := func(rowText, what string) {
		t.Helper()
		settle(what+" admitted", `(typeof window.laneValidation === 'string' && window.laneValidation.length > 0) || !!`+rowFailure(rowText))
		failIfRowFailed(rowText, what)
		id := testcheck.Must(artifact.ParseID(pageText(t, ctx, browser, `window.laneValidation`)))(t)
		assertBrowserPredicate(t, ctx, browser, `(() => { window.laneValidation = ''; return true; })()`)
		waitBrowserOperationState(t, workbench, id, operation.StateCompleted)
	}

	// 0. The cold page: no child, the Library is the way in, and the workbench's validation answers are captured.
	settle("cold page", `document.querySelector("#model-pill").textContent === "Choose a model" && !!document.querySelector("#cold-start") && window.overgo.errors.length === 0`)
	assertBrowserPredicate(t, ctx, browser, `(() => {
  window.laneValidation = '';
  const post = overgo.api.post;
  overgo.api.post = async function (path, body, options) {
    const result = await post.call(this, path, body, options);
    if (path === '/library/validate') window.laneValidation = result.operation;
    return result;
  };
  [...document.querySelectorAll("#cold-start button")].find((button) => button.textContent === "Open the Library").click();
  return true;
})()`)
	settle("library on the cold page", `!!document.querySelector("#panel-library.active input[placeholder='Search the Hugging Face hub']")`)

	// 1. The hub model in one action: search, then add, which downloads, registers and validates it
	// against the store behind the cold proxy.
	assertBrowserPredicate(t, ctx, browser, `(() => {
  const query = document.querySelector("#panel-library input[placeholder='Search the Hugging Face hub']");
  query.value = 'validated';
  [...document.querySelectorAll("#panel-library button")].find((button) => button.textContent === "Search").click();
  return true;
})()`)
	settle("hub search lists the lane repository", `[...document.querySelectorAll("#panel-library button")].some((button) => button.textContent === "Add")`)
	assertBrowserPredicate(t, ctx, browser, `(() => { [...document.querySelectorAll("#panel-library button")].find((button) => button.textContent === "Add").click(); return true; })()`)
	rowAnswers(laneHubRepository, "done", "the download")
	rowAnswers(laneHubRepository, "registered", "the downloaded model's registration")
	validated(laneHubRepository, "the downloaded model's validation")
	webuilane.Leg(t, "download leg", "one add downloaded, registered and validated the hub model through the cold proxy's workbench")

	// 2. A local model added with its projector keeps the projector as its projection candidate, and
	// its text path validates as an operation the workbench behind the shell runs.
	assertBrowserPredicate(t, ctx, browser, `(() => {
  document.querySelector("#panel-library input[placeholder='model GGUF or directory on disk']").value = `+strconv.Quote(projectedLocation)+`;
  document.querySelector("#panel-library input[placeholder='projector GGUF (optional)']").value = `+strconv.Quote(projectorPath)+`;
  [...document.querySelectorAll("#panel-library button")].find((button) => button.textContent === "Add a local model").click();
  return true;
})()`)
	rowAnswers(projectedLocation, "with projector", "the local model's registration with its projector")
	validated(projectedLocation, "the projector pair's validation")
	assertBrowserPredicate(t, ctx, browser, rowNote(projectedLocation, "projector"))
	webuilane.Leg(t, "projector leg", "%s validated its text path with projector %s registered", projectedName, filepath.Base(projectorPath))

	// 4. The validated download serves from the picker: the first child launches from the Library's page.
	name := filepath.Base(small)
	assertBrowserPredicate(t, ctx, browser, `(() => { history.replaceState(null, "", location.pathname); document.querySelector("#model-pill").click(); return true; })()`)
	settle("picker lists the validated model", `[...document.querySelectorAll(".topbar .card .row .mono")].some((node) => node.textContent === `+strconv.Quote(name)+`)`)
	assertBrowserPredicate(t, ctx, browser, `(() => {
  const row = [...document.querySelectorAll(".topbar .card .row")].find((node) => node.querySelector(".mono") && node.querySelector(".mono").textContent === `+strconv.Quote(name)+`);
  if (!row) return false;
  [...row.querySelectorAll("button")].find((button) => button.textContent === "Serve").click();
  return true;
})()`)
	settle("the chosen model serves", `document.querySelector("#model-pill").textContent === `+strconv.Quote(name)+` && !!document.querySelector("#panel-chat.active .composer textarea") && !window.overgo.modelSwitching()`)
	webuilane.Leg(t, "library validation leg", "downloaded, registered with a projector, validation admitted and %s served from the cold proxy", name)
}
