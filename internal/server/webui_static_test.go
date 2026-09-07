package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
)

func TestWebUIServesEmbeddedAssets(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	cases := []struct {
		path        string
		contentType string
		needle      string
	}{
		{"/", "text/html; charset=utf-8", "overgo"},
		{"/index.html", "text/html; charset=utf-8", "workbench"},
		{"/style.css", "text/css; charset=utf-8", "--acc"},
		{"/boot.js", "text/javascript; charset=utf-8", "window.overgo"},
		{"/viz.js", "text/javascript; charset=utf-8", "sparkline"},
		{"/md.js", "text/javascript; charset=utf-8", "overgo.md"},
		{"/workflow.js", "text/javascript; charset=utf-8", "workflowWorkspace"},
		{"/schema_form.js", "text/javascript; charset=utf-8", "schemaForm"},
		{"/mod/chat.js", "text/javascript; charset=utf-8", "/v1/responses"},
		{"/mod/agent.js", "text/javascript; charset=utf-8", "overgo.toolStep("},
		{"/mod/inbox.js", "text/javascript; charset=utf-8", "/operations/inbox"},
		{"/mod/generation.js", "text/javascript; charset=utf-8", `scope: "generation"`},
		{"/mod/discovery.js", "text/javascript; charset=utf-8", "/hub/search"},
		{"/mod/runtime.js", "text/javascript; charset=utf-8", "runtimeEvents"},
		{"/mod/datasets.js", "text/javascript; charset=utf-8", "/datasets"},
		{"/mod/training.js", "text/javascript; charset=utf-8", "/runs"},
		{"/mod/model_builder.js", "text/javascript; charset=utf-8", `scope: "model-builder"`},
		{"/mod/jobs.js", "text/javascript; charset=utf-8", "training-jobs"},
		{"/mod/recipe.js", "text/javascript; charset=utf-8", "/recipes/active"},
		{"/mod/compositions.js", "text/javascript; charset=utf-8", "/compositions/activate"},
		{"/mod/artifacts.js", "text/javascript; charset=utf-8", "/artifacts"},
		{"/mod/evaluations.js", "text/javascript; charset=utf-8", "/evaluations/capabilities"},
		{"/mod/automations.js", "text/javascript; charset=utf-8", "/automations/stream"},
		{"/mod/peers.js", "text/javascript; charset=utf-8", "/peers/stream"},
		{"/mod/analyze_model.js", "text/javascript; charset=utf-8", "/analyze/model"},
		{"/mod/analyze_vocab.js", "text/javascript; charset=utf-8", "/analyze/vocab"},
		{"/mod/analyze_logits.js", "text/javascript; charset=utf-8", "completion_probabilities"},
		{"/mod/analyze_states.js", "text/javascript; charset=utf-8", "/analyze/states"},
		{"/mod/analyze_attention.js", "text/javascript; charset=utf-8", "/analyze/attention"},
		{"/mod/analyze_tensors.js", "text/javascript; charset=utf-8", "/analyze/tensors"},
	}
	for _, testCase := range cases {
		response := serveTestRequest(handler, http.MethodGet, testCase.path, "")
		if response.Code != http.StatusOK {
			t.Fatalf("GET %s status = %d", testCase.path, response.Code)
		}
		if got := response.Header().Get("Content-Type"); got != testCase.contentType {
			t.Fatalf("GET %s content-type = %q, want %q", testCase.path, got, testCase.contentType)
		}
		if !strings.Contains(response.Body.String(), testCase.needle) {
			t.Fatalf("GET %s body missing %q", testCase.path, testCase.needle)
		}
	}
}

func TestEvaluationWorkbenchUsesDeclaredCapabilities(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	module := serveTestRequest(handler, http.MethodGet, "/mod/evaluations.js", "").Body.String()
	for _, token := range []string{
		"/evaluations/capabilities", "/evaluations/run", "/evaluations/history", "/evaluations/report",
		"/evaluations/failures", "/evaluations/compare", "overgo.waitOperation", "capability.suite.plan",
	} {
		if !strings.Contains(module, token) {
			t.Errorf("evaluation workbench missing %q", token)
		}
	}
	for _, benchmark := range []string{"mmlu", "truthfulqa", "ifeval", "bbh", "musr"} {
		if strings.Contains(strings.ToLower(module), benchmark) {
			t.Errorf("evaluation workbench embeds benchmark %q", benchmark)
		}
	}
	if !strings.Contains(serveTestRequest(handler, http.MethodGet, "/workspace/manifest", "").Body.String(), `"id":"evaluations"`) {
		t.Fatal("workbench shell does not load evaluation module")
	}
}

func TestWebUIRuntimeMonitor(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	get := func(path string) string {
		return serveTestRequest(handler, http.MethodGet, path, "").Body.String()
	}
	runtime := get("/mod/runtime.js")
	for _, token := range []string{"runtimeEvents.subscribe", "onActivate", "onDeactivate", "runtime.sessions", "runtime.activity"} {
		if !strings.Contains(runtime, token) {
			t.Errorf("runtime module missing %q", token)
		}
	}
	for _, polling := range []string{"overgo.poller", `api.get("/runtime/sessions"`, `api.get("/runtime/activity"`} {
		if strings.Contains(runtime, polling) {
			t.Errorf("runtime module retains polling path %q", polling)
		}
	}
	if strings.Contains(runtime, "include_text") {
		t.Error("runtime monitor requests retained text")
	}
	boot := get("/boot.js")
	for _, token := range []string{"function poller", "document.hidden", "t.onActivate", "t.onDeactivate"} {
		if !strings.Contains(boot, token) {
			t.Errorf("boot lifecycle missing %q", token)
		}
	}
	if !strings.Contains(get("/workspace/manifest"), `"id":"runtime"`) {
		t.Error("app shell does not load runtime module")
	}
}

func TestWorkflowStageGUI(t *testing.T) {
	runtime := serveTestRequest(newTestHandler(t, &fakeGenerator{}), http.MethodGet, "/mod/runtime.js", "").Body.String()
	for _, token := range []string{"data.stages", "Workflow stages", "stage.operation", "stage.attempt"} {
		if !strings.Contains(runtime, token) {
			t.Errorf("workflow stage GUI missing %q", token)
		}
	}
}

func TestToolDecisionGUI(t *testing.T) {
	runtime := serveTestRequest(newTestHandler(t, &fakeGenerator{}), http.MethodGet, "/mod/runtime.js", "").Body.String()
	for _, token := range []string{"overgo.decideOperation(", "item.recovery", "Grant ", "decline"} {
		if !strings.Contains(runtime, token) {
			t.Errorf("tool decision GUI missing %q", token)
		}
	}
}

func TestInteractionReplayGUI(t *testing.T) {
	runtime := serveTestRequest(newTestHandler(t, &fakeGenerator{}), http.MethodGet, "/mod/runtime.js", "").Body.String()
	for _, token := range []string{"data.interactions", "/interactions/replay?response=", "interaction.trace"} {
		if !strings.Contains(runtime, token) {
			t.Errorf("interaction replay GUI missing %q", token)
		}
	}
}

func TestRemoteAttemptGUI(t *testing.T) {
	runtime := serveTestRequest(newTestHandler(t, &fakeGenerator{}), http.MethodGet, "/mod/runtime.js", "").Body.String()
	if !strings.Contains(runtime, `item.compatibility ? " / peer"`) {
		t.Fatal("remote attempts are not identified from compatibility evidence")
	}
}

func TestAgentGUIUsesProjectedQueriesAndSSE(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	runtime := serveTestRequest(handler, http.MethodGet, "/mod/runtime.js", "").Body.String()
	workflow := serveTestRequest(handler, http.MethodGet, "/workflow.js", "").Body.String()
	for _, token := range []string{"runtimeEvents", "/runtime/activity/stream", "operation.snapshot"} {
		if !strings.Contains(workflow, token) {
			t.Errorf("shared runtime stream missing %q", token)
		}
	}
	for _, polling := range []string{"operationPollMilliseconds", `api.get("/operations?id="`, "overgo.poller"} {
		if strings.Contains(workflow, polling) || strings.Contains(runtime, polling) {
			t.Errorf("agent GUI retains polling path %q", polling)
		}
	}
}

func TestWebUIChatUsesServerContextAndTiming(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	const requestBody = `{"messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":1}`
	count := serveTestRequest(handler, http.MethodPost, "/v1/chat/completions/input_tokens", requestBody)
	var expected struct {
		InputTokens int `json:"input_tokens"`
	}
	if err := json.Unmarshal(count.Body.Bytes(), &expected); err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(requestBody),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("stream status = %d body=%s", response.Code, response.Body.String())
	}
	var terminal chatStreamResponse
	for line := range strings.SplitSeq(response.Body.String(), "\n") {
		if !strings.HasPrefix(line, "data: ") || strings.HasSuffix(line, "[DONE]") {
			continue
		}
		var chunk chatStreamResponse
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
			t.Fatal(err)
		}
		if chunk.Usage != nil {
			terminal = chunk
		}
	}
	if terminal.Usage == nil || terminal.Timings == nil {
		t.Fatalf("terminal stream lacks facts: %s", response.Body.String())
	}
	if terminal.Usage.PromptTokens != expected.InputTokens ||
		uint64(terminal.Usage.CompletionTokens) != terminal.Timings.PredictedN {
		t.Fatalf("stream facts = usage %+v timings %+v count %d", terminal.Usage, terminal.Timings, expected.InputTokens)
	}

	get := func(path string) string {
		return serveTestRequest(handler, http.MethodGet, path, "").Body.String()
	}
	chat := get("/mod/chat.js")
	for _, token := range []string{
		"overgo.capabilities(",
		`api.post("/v1/responses/input_tokens"`,
		"overgo.api.stream(",
		"Context ratio",
		"terminal.usage",
		"terminal.timings",
	} {
		if !strings.Contains(chat, token) {
			t.Errorf("chat module missing %q", token)
		}
	}
	for _, embeddedDefault := range []string{`value: "0.7"`, `value: "512"`, `|| 512`} {
		if strings.Contains(chat, embeddedDefault) {
			t.Errorf("chat module embeds sampling default %q", embeddedDefault)
		}
	}
	if !strings.Contains(get("/boot.js"), "async stream(path, body, opts)") {
		t.Error("shared API client does not own streaming fetch")
	}
}

func TestWebUIUnknownPathReturns404(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	response := serveTestRequest(handler, http.MethodGet, "/no/such/asset.js", "")
	if response.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", response.Code)
	}
	if !strings.Contains(response.Body.String(), "route not found") {
		t.Fatalf("body = %s", response.Body.String())
	}
}

// The GUI shell and its assets must load without a bearer token even when the
// server is key-protected — otherwise the page could never present the key
// field. API routes stay protected; only the static assets are public.
func TestWebUIAssetsPublicWhenAPIKeyConfigured(t *testing.T) {
	handler, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, APIKey: "secret"}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	asset := serveTestRequest(handler, http.MethodGet, "/", "")
	if asset.Code != http.StatusOK {
		t.Fatalf("GET / with key configured = %d, want 200 (assets public)", asset.Code)
	}
	// A protected API route with no bearer must still be rejected.
	protected := serveTestRequest(handler, http.MethodGet, "/analyze/model", "")
	if protected.Code != http.StatusUnauthorized {
		t.Fatalf("GET /analyze/model without bearer = %d, want 401", protected.Code)
	}
}

// TestWebUIStyleInvariants guards two concrete regressions: (1) client CSS
// custom properties must reference names the palette actually defines — an
// undefined var silently renders nothing (the energy-entropy bar filled with a
// non-existent --accent); (2) the panel must own its horizontal overflow so a
// wide table cannot force the whole page body to scroll sideways.
func TestWebUIStyleInvariants(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})

	// Every var(--name) used in a module must be a name style.css defines.
	css := serveTestRequest(handler, http.MethodGet, "/style.css", "").Body.String()
	defined := map[string]bool{}
	for _, m := range regexp.MustCompile(`--[a-z0-9-]+\s*:`).FindAllString(css, -1) {
		defined[strings.TrimSpace(strings.TrimSuffix(m, ":"))] = true
	}
	for _, asset := range []string{"/mod/analyze_tensors.js", "/viz.js", "/mod/analyze_states.js", "/mod/analyze_attention.js"} {
		body := serveTestRequest(handler, http.MethodGet, asset, "").Body.String()
		for _, ref := range regexp.MustCompile(`var\(\s*(--[a-z0-9-]+)`).FindAllStringSubmatch(body, -1) {
			if !defined[ref[1]] {
				t.Errorf("%s references undefined CSS var %q", asset, ref[1])
			}
		}
	}

	// Panels scroll their own overflow so wide tables never scroll the body.
	if !strings.Contains(css, "overflow-x:auto") {
		t.Error("style.css: expected a panel overflow-x:auto rule so wide tables scroll in-panel")
	}
}

// TestWebUIAuthUX guards the centralized 401 handling: a shared friendlyError
// helper directing the user to connection settings, and the analysis tabs
// routing their errors through the helper instead of leaking the raw bearer
// error.
func TestWebUIAuthUX(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	get := func(p string) string { return serveTestRequest(handler, http.MethodGet, p, "").Body.String() }

	boot := get("/boot.js")
	for _, needle := range []string{"friendlyError", "Settings", "Connection"} {
		if !strings.Contains(boot, needle) {
			t.Errorf("boot.js missing %q", needle)
		}
	}
	page := get("/")
	for _, needle := range []string{`id="settings-dialog"`, `id="api-key"`} {
		if !strings.Contains(page, needle) {
			t.Errorf("settings missing %q", needle)
		}
	}
	for _, asset := range []string{"/mod/analyze_model.js", "/mod/analyze_vocab.js", "/mod/analyze_tensors.js"} {
		if !strings.Contains(get(asset), "friendlyError") {
			t.Errorf("%s does not route errors through friendlyError", asset)
		}
	}
}

// TestWebUIHeatmapLegend guards the heatmap readability additions: a color-scale
// legend built from the same ramp() the cells use, and axis tick labels driven by
// caller-supplied labels (so a reader can map color→value and read the axes
// instead of hovering every cell).
func TestWebUIHeatmapLegend(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	viz := serveTestRequest(handler, http.MethodGet, "/viz.js", "").Body.String()
	for _, needle := range []string{"colorScaleLegend", "linearGradient", "rowLabels", "colLabels"} {
		if !strings.Contains(viz, needle) {
			t.Errorf("viz.js heatmap missing %q", needle)
		}
	}
	for _, asset := range []string{"/mod/analyze_attention.js", "/mod/analyze_states.js"} {
		if !strings.Contains(serveTestRequest(handler, http.MethodGet, asset, "").Body.String(), "labels:") {
			t.Errorf("%s does not pass labels to heatmap (axis ticks)", asset)
		}
	}
}

func TestSignedSeriesPreservesNegativeMeasurements(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	viz := serveTestRequest(handler, http.MethodGet, "/viz.js", "").Body.String()
	start := strings.Index(viz, "function signedSeries")
	if start < 0 {
		t.Fatal("signed series primitive absent")
	}
	end := strings.Index(viz[start:], "function probBars")
	if end < 0 {
		t.Fatal("signed series boundary absent")
	}
	series := viz[start : start+end]
	for _, token := range []string{"Math.min(0, ...values)", `class: "zero-axis"`, `"data-value": String(value)`} {
		if !strings.Contains(series, token) {
			t.Errorf("signed series missing %q", token)
		}
	}
	if strings.Contains(series, "Math.max(0, value)") {
		t.Error("signed series clamps negative measurements")
	}
}

func TestRLWorkspaceRendersMeasuredEvidence(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	training := serveTestRequest(handler, http.MethodGet, "/mod/training.js", "").Body.String()
	for _, token := range []string{
		"DPO loss", "GRPO loss", "Evaluator reward", "Margin decomposition", "Chosen / rejected pair", "Optimizer health",
		"Checkpoint comparison", "policy_margin", "reference_margin", "relative_margin",
		"mean_reward", "reward_dispersion", "gradient_l2", "update_l2", "/artifacts/content?id=",
	} {
		if !strings.Contains(training, token) {
			t.Errorf("training evidence view missing %q", token)
		}
	}
	for _, forbidden := range []string{"smooth", "movingAverage"} {
		if strings.Contains(training, forbidden) {
			t.Errorf("training evidence view contains derived signal %q", forbidden)
		}
	}
}

func TestRLWorkspaceUsesGenericWorkflowEndpoints(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	workflow := serveTestRequest(handler, http.MethodGet, "/workflow.js", "").Body.String()
	jobs := serveTestRequest(handler, http.MethodGet, "/mod/jobs.js", "").Body.String()
	for _, token := range []string{`"/" + definition.scope + "/run"`, "overgo.waitOperation", "/operations/cancel"} {
		if !strings.Contains(workflow, token) {
			t.Errorf("generic workflow missing %q", token)
		}
	}
	if !strings.Contains(jobs, `scope: "training"`) || !strings.Contains(jobs, "trainingEvidence.renderOperation") {
		t.Error("training registration does not use the generic workflow renderer")
	}
	if strings.Contains(workflow, "/dpo") || strings.Contains(jobs, "/dpo") {
		t.Error("RL workspace adds a DPO-only transport")
	}
}

// TestWebUIShellCache guards the shared /analyze/model cache (fetched once for
// capability gating, the Model tab, and the lens vocab size) and the periodic
// health re-probe so a dropped/restored server updates the status pill.
func TestWebUIShellCache(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	get := func(p string) string { return serveTestRequest(handler, http.MethodGet, p, "").Body.String() }
	boot := get("/boot.js")
	for _, needle := range []string{"modelInfo", "invalidateModel", "setInterval(refreshStatus"} {
		if !strings.Contains(boot, needle) {
			t.Errorf("boot.js missing %q", needle)
		}
	}
	for _, asset := range []string{"/mod/analyze_model.js", "/mod/analyze_logits.js"} {
		body := get(asset)
		if !strings.Contains(body, "modelInfo") {
			t.Errorf("%s does not use the shared modelInfo cache", asset)
		}
		if strings.Contains(body, `api.get("/analyze/model")`) {
			t.Errorf("%s still fetches /analyze/model directly (bypasses the cache)", asset)
		}
	}
}

// TestWebUICancel guards the cancelable-run path for the tabs that drive a real
// forward pass (lens, hidden-states, attention). The abort semantics live once
// in the shared runner; each tab drives it and forwards the signal.
func TestWebUICancel(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	get := func(p string) string { return serveTestRequest(handler, http.MethodGet, p, "").Body.String() }
	boot := get("/boot.js")
	for _, needle := range []string{"AbortController", "controller.abort", "AbortError", "function runner"} {
		if !strings.Contains(boot, needle) {
			t.Errorf("boot.js shared runner missing %q", needle)
		}
	}
	for _, asset := range []string{"/mod/analyze_logits.js", "/mod/analyze_states.js", "/mod/analyze_attention.js"} {
		body := get(asset)
		if !strings.Contains(body, "overgo.runner(") && !strings.Contains(body, "overgo.analysisSurface(") {
			t.Errorf("%s does not use the shared overgo.runner", asset)
		}
		if !strings.Contains(body, "{ signal }") {
			t.Errorf("%s does not forward the abort signal to its request", asset)
		}
	}
}

// TestWebUIClientDedup guards the tightening: displayToken is defined once in the
// shell and the token-showing tabs use it instead of each redefining it.
func TestWebUIClientDedup(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	get := func(p string) string { return serveTestRequest(handler, http.MethodGet, p, "").Body.String() }
	if !strings.Contains(get("/boot.js"), "function displayToken") {
		t.Error("boot.js should define the shared displayToken")
	}
	for _, asset := range []string{"/mod/analyze_logits.js", "/mod/analyze_states.js", "/mod/analyze_attention.js"} {
		if strings.Contains(get(asset), "function displayToken") {
			t.Errorf("%s redefines displayToken instead of using the shared overgo.displayToken", asset)
		}
	}
}

// TestWebUIChatMarkdown guards the chat markdown renderer: it must build DOM
// (never innerHTML — model output is untrusted), scheme-check link hrefs, offer
// copy buttons, and be wired into the chat tab and the shell load order.
func TestWebUIChatMarkdown(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	get := func(p string) string { return serveTestRequest(handler, http.MethodGet, p, "").Body.String() }

	md := get("/md.js")
	if strings.Contains(md, ".innerHTML") {
		t.Error("md.js assigns .innerHTML — model output must render as DOM text, never markup")
	}
	for _, needle := range []string{"md-codeblock", "clipboard.writeText", "https?:", "createTextNode"} {
		if !strings.Contains(md, needle) {
			t.Errorf("md.js missing %q", needle)
		}
	}
	// The thread renderer in composer.js owns assistant markdown and copy for
	// every surface; chat.js reaches it through the shared thread.
	composer := get("/composer.js")
	for _, needle := range []string{"overgo.md(", "overgo.copyButton"} {
		if !strings.Contains(composer, needle) {
			t.Errorf("composer.js does not use %q", needle)
		}
	}
	if !strings.Contains(get("/mod/chat.js"), "overgo.thread(") {
		t.Error("chat.js does not render through the shared thread")
	}
	// md.js must load before any module so overgo.md exists when chat renders:
	// the loader lists it among the libraries it awaits before the modules.
	boot := get("/boot.js")
	if strings.Index(boot, `"/md.js"`) < 0 || strings.Index(boot, `"/md.js"`) > strings.Index(boot, "loadWorkspaceModules(") {
		t.Error("boot.js must list /md.js among the libraries loaded before the modules")
	}
}

// TestWebUIContentSecurityPolicy guards the strict CSP: same-origin scripts only
// (no inline), nosniff, and no framing. The landing page must therefore carry no
// inline <script> or inline event handler — those moved to probe.js.
func TestWebUIContentSecurityPolicy(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, p := range []string{"/", "/index.html", "/app.html", "/boot.js"} {
		resp := serveTestRequest(handler, http.MethodGet, p, "")
		csp := resp.Header().Get("Content-Security-Policy")
		if !strings.Contains(csp, "script-src 'self';") {
			t.Errorf("GET %s CSP script-src not strict-self: %q", p, csp)
		}
		if strings.Contains(csp, "script-src 'self' 'unsafe-inline'") {
			t.Errorf("GET %s CSP allows inline script: %q", p, csp)
		}
		for _, directive := range []string{"object-src 'none'", "frame-ancestors 'none'", "connect-src 'self'", "img-src 'self' data: blob:", "media-src 'self' data: blob:"} {
			if !strings.Contains(csp, directive) {
				t.Errorf("GET %s CSP missing %q", p, directive)
			}
		}
		if resp.Header().Get("X-Content-Type-Options") != "nosniff" {
			t.Errorf("GET %s missing X-Content-Type-Options: nosniff", p)
		}
	}
	index := serveTestRequest(handler, http.MethodGet, "/index.html", "").Body.String()
	if strings.Contains(index, "<script>") {
		t.Error("index.html has an inline <script> — blocked by script-src 'self'")
	}
	if strings.Contains(index, "onclick=") {
		t.Error("index.html has an inline onclick — blocked by script-src 'self'")
	}
}

func TestWebUIRejectsNonGet(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/", nil))
	if response.Code != http.StatusNotFound {
		t.Fatalf("POST / status = %d, want 404", response.Code)
	}
}

// TestCollapsibleSections pins the fold contract: the shell serves the
// shared fold helper, the dense pages consume it instead of stacking
// section headers, and the stylesheet carries the disclosure control
// styling -- so optional free-entry panels collapse to their headers.
func TestCollapsibleSections(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	assertions := []struct {
		path, needle string
	}{
		{"/boot.js", "function fold("},
		{"/mod/agent.js", "overgo.fold("},
		{"/mod/runtime.js", "overgo.fold("},
		{"/style.css", ".fold[open] > summary::before"},
		// The agent page leads with the task: simple creation posts to
		// the derive-everything endpoint, the session names itself, and
		// every operator panel folds behind an Advanced header.
		{"/mod/agent.js", "/agents/create"},
		{"/mod/agent.js", "Advanced: manual tool steps"},
		{"/mod/agent.js", "autoSession"},
	}
	for _, assertion := range assertions {
		response := serveTestRequest(handler, http.MethodGet, assertion.path, "")
		if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), assertion.needle) {
			t.Fatalf("%s status=%d missing %q", assertion.path, response.Code, assertion.needle)
		}
	}
}
