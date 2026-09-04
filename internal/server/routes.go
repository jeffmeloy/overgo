package server

import (
	"net/http"
	"slices"
	"strings"

	"overgo/internal/apimanifest"
)

type routeAuthentication string

const (
	routePublic routeAuthentication = "public"
	routeBearer routeAuthentication = "bearer"
)

type routeHandler func(*Handler, http.ResponseWriter, *http.Request)

type workflowRouteAction string

const (
	workflowCapabilitiesAction workflowRouteAction = "capabilities"
	workflowRunAction          workflowRouteAction = "run"
)

type routeDescriptor struct {
	Path           string
	Authentication routeAuthentication
	Methods        []string
	Handler        routeHandler
	Workflow       WorkflowKind
	WorkflowAction workflowRouteAction
}

var routeCatalog = []routeDescriptor{
	{Path: "/health", Authentication: routePublic, Methods: []string{http.MethodGet}, Handler: (*Handler).health},
	{Path: "/healthz", Authentication: routePublic, Methods: []string{http.MethodGet}, Handler: (*Handler).health},
	{Path: "/v1/health", Authentication: routePublic, Methods: []string{http.MethodGet}, Handler: (*Handler).health},
	{Path: "/metrics", Authentication: routePublic, Methods: []string{http.MethodGet}, Handler: (*Handler).metrics},
	{Path: "/automations/webhook", Authentication: routePublic, Methods: []string{http.MethodPost}, Handler: (*Handler).automationWebhook},
	{Path: "/v1/models", Authentication: routePublic, Methods: []string{http.MethodGet}, Handler: (*Handler).models},
	{Path: "/models", Authentication: routePublic, Methods: []string{http.MethodGet}, Handler: (*Handler).models},
	{Path: "/v1/completions", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).completions},
	{Path: "/completion", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).nativeCompletions},
	{Path: "/completions", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).nativeCompletions},
	{Path: "/infill", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).infill},
	{Path: "/v1/chat/completions", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).chatCompletions},
	{Path: "/chat/completions", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).chatCompletions},
	{Path: "/responses", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).responses},
	{Path: "/v1/responses", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).responses},
	{Path: "/chat/completions/input_tokens", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).chatInputTokens},
	{Path: "/v1/chat/completions/input_tokens", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).chatInputTokens},
	{Path: "/responses/input_tokens", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).responsesInputTokens},
	{Path: "/v1/responses/input_tokens", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).responsesInputTokens},
	{Path: "/v1/messages/count_tokens", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).anthropicInputTokens},
	{Path: "/v1/messages", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).anthropicMessages},
	{Path: "/v1/embeddings", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).embeddings},
	{Path: "/v1/images/generations", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).nativeImageGeneration},
	{Path: "/v1/videos/generations", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).nativeVideoGeneration},
	{Path: "/v1/videos/edits", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).nativeVideoEdit},
	{Path: "/v1/audio/speech", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).nativeAudioSpeech},
	{Path: "/embedding", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).nativeEmbeddings},
	{Path: "/embeddings", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).nativeEmbeddings},
	{Path: "/rerank", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).rerank},
	{Path: "/reranking", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).rerank},
	{Path: "/v1/rerank", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).rerank},
	{Path: "/v1/reranking", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).rerank},
	{Path: "/apply-template", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).applyTemplate},
	{Path: "/tokenize", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).tokenize},
	{Path: "/detokenize", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).detokenize},
	{Path: "/props", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).properties},
	{Path: "/analyze/model", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).analyzeModel},
	{Path: "/analyze/vocab", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).analyzeVocab},
	{Path: "/analyze/states", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).analyzeStates},
	{Path: "/analyze/attention", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).analyzeAttention},
	{Path: "/analyze/tensors", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).analyzeTensors},
	{Path: "/analyze/tensors/similar", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).analyzeTensorsSimilar},
	{Path: "/datasets", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).browseDatasets},
	{Path: "/datasets/preview", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).previewDataset},
	{Path: "/runs", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).browseRuns},
	{Path: "/store/deltas", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).storeDeltas},
	{Path: "/interactions/replay", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).interactionReplay},
	{Path: "/capabilities/bundles", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).capabilityBundles},
	{Path: "/recipes/active", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).activeRecipe},
	{Path: "/compositions", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).compositionInventory},
	{Path: "/compositions/activate", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).activateComposition},
	{Path: "/compositions/generate", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).compositeGeneration},
	{Path: "/operations", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).operationStatus},
	{Path: "/operations/inbox", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).operationInbox},
	{Path: "/operations/decisions", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).operatorDecisions},
	{Path: "/operations/timeline", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).operatorTimeline},
	{Path: "/operations/dag", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).operationDAG},
	{Path: "/operations/cancel", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).operationCancel},
	{Path: "/operations/decision", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).operationDecision},
	{Path: "/operations/wait", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).operationWait},
	{Path: "/evaluations/capabilities", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).evaluationCapabilities},
	{Path: "/evaluations/run", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).evaluationRun},
	{Path: "/evaluations/history", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).evaluationHistory},
	{Path: "/evaluations/report", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).evaluationReport},
	{Path: "/evaluations/failures", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).evaluationFailures},
	{Path: "/evaluations/compare", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).evaluationCompare},
	{Path: "/generation/capabilities", Authentication: routeBearer, Methods: []string{http.MethodGet}, Workflow: WorkflowGeneration, WorkflowAction: workflowCapabilitiesAction},
	{Path: "/generation/run", Authentication: routeBearer, Methods: []string{http.MethodPost}, Workflow: WorkflowGeneration, WorkflowAction: workflowRunAction},
	{Path: "/training/capabilities", Authentication: routeBearer, Methods: []string{http.MethodGet}, Workflow: WorkflowTraining, WorkflowAction: workflowCapabilitiesAction},
	{Path: "/training/run", Authentication: routeBearer, Methods: []string{http.MethodPost}, Workflow: WorkflowTraining, WorkflowAction: workflowRunAction},
	{Path: "/model-builder/capabilities", Authentication: routeBearer, Methods: []string{http.MethodGet}, Workflow: WorkflowModelBuild, WorkflowAction: workflowCapabilitiesAction},
	{Path: "/model-builder/run", Authentication: routeBearer, Methods: []string{http.MethodPost}, Workflow: WorkflowModelBuild, WorkflowAction: workflowRunAction},
	{Path: "/export/capabilities", Authentication: routeBearer, Methods: []string{http.MethodGet}, Workflow: WorkflowExport, WorkflowAction: workflowCapabilitiesAction},
	{Path: "/export/run", Authentication: routeBearer, Methods: []string{http.MethodPost}, Workflow: WorkflowExport, WorkflowAction: workflowRunAction},
	{Path: "/artifacts", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).artifactGallery},
	{Path: "/artifacts/content", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).artifactContent},
	{Path: "/runtime/sessions", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).runtimeSessions},
	{Path: "/runtime/activity", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).runtimeActivity},
	{Path: "/runtime/activity/stream", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).runtimeActivityStream},
	{Path: "/runtime/peers", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).remotePeerAuthority},
	{Path: "/slots", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).slotStatus},
	{Path: "/lora-adapters", Authentication: routeBearer, Methods: []string{http.MethodGet, http.MethodPost}, Handler: (*Handler).loraAdapters},
	{Path: "/catalog/models", Authentication: routePublic, Methods: []string{http.MethodGet}, Handler: (*Handler).catalogModels},
	// Hub search reaches outward and downloads mutate local disk; agent
	// routes execute tools and expose session transcripts. All of them
	// require the bearer credential -- only local read-only catalogs, the
	// health/metrics probes, and the signature-verified automation webhook
	// stay public.
	{Path: "/hub/search", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).hubSearch},
	{Path: "/hub/downloads", Authentication: routeBearer, Methods: []string{http.MethodGet, http.MethodPost, http.MethodDelete}, Handler: (*Handler).hubDownloads},
	{Path: "/agent/tools", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).agentTools},
	{Path: "/agent/step", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).agentStep},
	{Path: "/agent/approval", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).agentApprovalPreview},
	{Path: "/agent/provenance", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).agentProvenance},
	{Path: "/agent/sessions", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).agentSessionList},
	// The workbench's own routes: the manifest, schema and route table are
	// public like the shell they feed; the operations evidence and the
	// automation, peer and agent workspaces take the bearer credential. They
	// were a prefix chain in front of the asset handler with five copies of
	// the authorization block; the table owns them like every other route.
	{Path: "/workspace/manifest", Authentication: routePublic, Methods: []string{http.MethodGet}, Handler: (*Handler).workspaceManifest},
	{Path: "/workspace/schema", Authentication: routePublic, Methods: []string{http.MethodGet}, Handler: (*Handler).workspaceSchema},
	{Path: "/workspace/routes", Authentication: routePublic, Methods: []string{http.MethodGet}, Handler: (*Handler).workspaceRoutes},
	{Path: "/interactions", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).conversations},
	{Path: "/interactions/messages", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).conversationMessages},
	{Path: "/interactions/label", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).conversationLabel},
	{Path: "/interactions/follow", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).conversationFollow},
	{Path: "/operations/evidence", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).operationEvidence},
	{Path: "/automations", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).automationWorkspace},
	{Path: "/automations/definitions", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).automationWorkspace},
	{Path: "/automations/activate", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).automationWorkspace},
	{Path: "/automations/run", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).automationWorkspace},
	{Path: "/automations/schedule", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).automationWorkspace},
	{Path: "/automations/history", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).automationWorkspace},
	{Path: "/automations/stream", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).automationWorkspace},
	{Path: "/peers", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).peerWorkspace},
	{Path: "/peers/enroll", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).peerWorkspace},
	{Path: "/peers/capability", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).peerWorkspace},
	{Path: "/peers/heartbeat", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).peerWorkspace},
	{Path: "/peers/state", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).peerWorkspace},
	{Path: "/peers/placement", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).peerWorkspace},
	{Path: "/peers/reconcile", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).peerWorkspace},
	{Path: "/peers/evidence", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).peerWorkspace},
	{Path: "/peers/stream", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).peerWorkspace},
	{Path: "/agents", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).agentControl},
	{Path: "/agents/create", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).agentControl},
	{Path: "/agents/definitions", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).agentControl},
	{Path: "/agents/activate", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).agentControl},
	{Path: "/agents/state", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).agentControl},
	{Path: "/agents/step", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).agentControl},
	{Path: "/agents/approval", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).agentControl},
	{Path: "/agents/chat", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).agentControl},
	{Path: "/agents/retrieval", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).agentControl},
	{Path: "/agents/automation", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).agentControl},
	{Path: "/agents/evidence", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).agentControl},
	{Path: "/agents/stream", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).agentControl},
}

var routesByPath = compileRouteIndex(routeCatalog)
var versionedRouteFallback = routeDescriptor{Path: "/v1/", Authentication: routeBearer, Handler: (*Handler).serveWebUI}

// APIManifestRoutes projects the runtime route authority into release metadata.
func APIManifestRoutes() []apimanifest.Route {
	routes := make([]apimanifest.Route, 0, len(routeCatalog))
	for _, route := range routeCatalog {
		for _, method := range route.Methods {
			routes = append(routes, apimanifest.Route{
				Path: route.Path, Method: method, Authentication: string(route.Authentication),
			})
		}
	}
	return routes
}

func compileRouteIndex(routes []routeDescriptor) map[string]routeDescriptor {
	index := make(map[string]routeDescriptor, len(routes))
	for _, route := range routes {
		workflow := route.Workflow != "" && route.WorkflowAction != ""
		if route.Path == "" || len(route.Methods) == 0 || (route.Handler == nil) == !workflow || index[route.Path].Path != "" {
			panic("server: invalid or duplicate route")
		}
		index[route.Path] = route
	}
	return index
}

func resolveRoute(path string) (routeDescriptor, bool) {
	if route, found := routesByPath[path]; found {
		return route, true
	}
	return versionedRouteFallback, strings.HasPrefix(path, versionedRouteFallback.Path)
}

func (r routeDescriptor) serve(h *Handler, response http.ResponseWriter, request *http.Request) {
	if r.Handler != nil {
		r.Handler(h, response, request)
		return
	}
	if r.WorkflowAction == workflowCapabilitiesAction {
		h.workflowCapabilities(response, request, r.Workflow)
		return
	}
	h.workflowRun(response, request, r.Workflow)
}

func (r routeDescriptor) accepts(method string) bool {
	return slices.Contains(r.Methods, method)
}
