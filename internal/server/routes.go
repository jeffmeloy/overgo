package server

import (
	"net/http"
	"strings"
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
	{Path: "/interactions/replay", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).interactionReplay},
	{Path: "/capabilities/bundles", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).capabilityBundles},
	{Path: "/recipes/active", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).activeRecipe},
	{Path: "/compositions", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).compositionInventory},
	{Path: "/compositions/activate", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).activateComposition},
	{Path: "/compositions/generate", Authentication: routeBearer, Methods: []string{http.MethodPost}, Handler: (*Handler).compositeGeneration},
	{Path: "/operations", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).operationStatus},
	{Path: "/operations/inbox", Authentication: routeBearer, Methods: []string{http.MethodGet}, Handler: (*Handler).operationInbox},
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
	{Path: "/hub/search", Authentication: routePublic, Methods: []string{http.MethodGet}, Handler: (*Handler).hubSearch},
	{Path: "/hub/downloads", Authentication: routePublic, Methods: []string{http.MethodGet, http.MethodPost, http.MethodDelete}, Handler: (*Handler).hubDownloads},
	{Path: "/agent/tools", Authentication: routePublic, Methods: []string{http.MethodGet}, Handler: (*Handler).agentTools},
	{Path: "/agent/step", Authentication: routePublic, Methods: []string{http.MethodPost}, Handler: (*Handler).agentStep},
	{Path: "/agent/approval", Authentication: routePublic, Methods: []string{http.MethodPost}, Handler: (*Handler).agentApprovalPreview},
	{Path: "/agent/provenance", Authentication: routePublic, Methods: []string{http.MethodGet}, Handler: (*Handler).agentProvenance},
	{Path: "/agent/sessions", Authentication: routePublic, Methods: []string{http.MethodGet}, Handler: (*Handler).agentSessionList},
}

var routesByPath = compileRouteIndex(routeCatalog)
var versionedRouteFallback = routeDescriptor{Path: "/v1/", Authentication: routeBearer, Handler: (*Handler).serveWebUI}

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
	for _, accepted := range r.Methods {
		if method == accepted {
			return true
		}
	}
	return false
}
