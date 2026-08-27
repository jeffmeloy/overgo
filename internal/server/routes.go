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
	Handler        routeHandler
	Workflow       WorkflowKind
	WorkflowAction workflowRouteAction
}

var routeCatalog = []routeDescriptor{
	{Path: "/health", Authentication: routePublic, Handler: (*Handler).health},
	{Path: "/healthz", Authentication: routePublic, Handler: (*Handler).health},
	{Path: "/v1/health", Authentication: routePublic, Handler: (*Handler).health},
	{Path: "/metrics", Authentication: routePublic, Handler: (*Handler).metrics},
	{Path: "/v1/models", Authentication: routePublic, Handler: (*Handler).models},
	{Path: "/models", Authentication: routePublic, Handler: (*Handler).models},
	{Path: "/v1/completions", Authentication: routeBearer, Handler: (*Handler).completions},
	{Path: "/completion", Authentication: routeBearer, Handler: (*Handler).nativeCompletions},
	{Path: "/completions", Authentication: routeBearer, Handler: (*Handler).nativeCompletions},
	{Path: "/infill", Authentication: routeBearer, Handler: (*Handler).infill},
	{Path: "/v1/chat/completions", Authentication: routeBearer, Handler: (*Handler).chatCompletions},
	{Path: "/chat/completions", Authentication: routeBearer, Handler: (*Handler).chatCompletions},
	{Path: "/responses", Authentication: routeBearer, Handler: (*Handler).responses},
	{Path: "/v1/responses", Authentication: routeBearer, Handler: (*Handler).responses},
	{Path: "/chat/completions/input_tokens", Authentication: routeBearer, Handler: (*Handler).chatInputTokens},
	{Path: "/v1/chat/completions/input_tokens", Authentication: routeBearer, Handler: (*Handler).chatInputTokens},
	{Path: "/responses/input_tokens", Authentication: routeBearer, Handler: (*Handler).responsesInputTokens},
	{Path: "/v1/responses/input_tokens", Authentication: routeBearer, Handler: (*Handler).responsesInputTokens},
	{Path: "/v1/messages/count_tokens", Authentication: routeBearer, Handler: (*Handler).anthropicInputTokens},
	{Path: "/v1/messages", Authentication: routeBearer, Handler: (*Handler).anthropicMessages},
	{Path: "/v1/embeddings", Authentication: routeBearer, Handler: (*Handler).embeddings},
	{Path: "/v1/images/generations", Authentication: routeBearer, Handler: (*Handler).nativeImageGeneration},
	{Path: "/v1/videos/generations", Authentication: routeBearer, Handler: (*Handler).nativeVideoGeneration},
	{Path: "/v1/videos/edits", Authentication: routeBearer, Handler: (*Handler).nativeVideoEdit},
	{Path: "/v1/audio/speech", Authentication: routeBearer, Handler: (*Handler).nativeAudioSpeech},
	{Path: "/embedding", Authentication: routeBearer, Handler: (*Handler).nativeEmbeddings},
	{Path: "/embeddings", Authentication: routeBearer, Handler: (*Handler).nativeEmbeddings},
	{Path: "/rerank", Authentication: routeBearer, Handler: (*Handler).rerank},
	{Path: "/reranking", Authentication: routeBearer, Handler: (*Handler).rerank},
	{Path: "/v1/rerank", Authentication: routeBearer, Handler: (*Handler).rerank},
	{Path: "/v1/reranking", Authentication: routeBearer, Handler: (*Handler).rerank},
	{Path: "/apply-template", Authentication: routeBearer, Handler: (*Handler).applyTemplate},
	{Path: "/tokenize", Authentication: routeBearer, Handler: (*Handler).tokenize},
	{Path: "/detokenize", Authentication: routeBearer, Handler: (*Handler).detokenize},
	{Path: "/props", Authentication: routeBearer, Handler: (*Handler).properties},
	{Path: "/analyze/model", Authentication: routeBearer, Handler: (*Handler).analyzeModel},
	{Path: "/analyze/vocab", Authentication: routeBearer, Handler: (*Handler).analyzeVocab},
	{Path: "/analyze/states", Authentication: routeBearer, Handler: (*Handler).analyzeStates},
	{Path: "/analyze/attention", Authentication: routeBearer, Handler: (*Handler).analyzeAttention},
	{Path: "/analyze/tensors", Authentication: routeBearer, Handler: (*Handler).analyzeTensors},
	{Path: "/analyze/tensors/similar", Authentication: routeBearer, Handler: (*Handler).analyzeTensorsSimilar},
	{Path: "/datasets", Authentication: routeBearer, Handler: (*Handler).browseDatasets},
	{Path: "/datasets/preview", Authentication: routeBearer, Handler: (*Handler).previewDataset},
	{Path: "/runs", Authentication: routeBearer, Handler: (*Handler).browseRuns},
	{Path: "/interactions/replay", Authentication: routeBearer, Handler: (*Handler).interactionReplay},
	{Path: "/capabilities/bundles", Authentication: routeBearer, Handler: (*Handler).capabilityBundles},
	{Path: "/recipes/active", Authentication: routeBearer, Handler: (*Handler).activeRecipe},
	{Path: "/compositions", Authentication: routeBearer, Handler: (*Handler).compositionInventory},
	{Path: "/compositions/activate", Authentication: routeBearer, Handler: (*Handler).activateComposition},
	{Path: "/compositions/generate", Authentication: routeBearer, Handler: (*Handler).compositeGeneration},
	{Path: "/operations", Authentication: routeBearer, Handler: (*Handler).operationStatus},
	{Path: "/operations/inbox", Authentication: routeBearer, Handler: (*Handler).operationInbox},
	{Path: "/operations/dag", Authentication: routeBearer, Handler: (*Handler).operationDAG},
	{Path: "/operations/cancel", Authentication: routeBearer, Handler: (*Handler).operationCancel},
	{Path: "/operations/decision", Authentication: routeBearer, Handler: (*Handler).operationDecision},
	{Path: "/operations/wait", Authentication: routeBearer, Handler: (*Handler).operationWait},
	{Path: "/evaluations/capabilities", Authentication: routeBearer, Handler: (*Handler).evaluationCapabilities},
	{Path: "/evaluations/run", Authentication: routeBearer, Handler: (*Handler).evaluationRun},
	{Path: "/evaluations/history", Authentication: routeBearer, Handler: (*Handler).evaluationHistory},
	{Path: "/evaluations/report", Authentication: routeBearer, Handler: (*Handler).evaluationReport},
	{Path: "/evaluations/failures", Authentication: routeBearer, Handler: (*Handler).evaluationFailures},
	{Path: "/evaluations/compare", Authentication: routeBearer, Handler: (*Handler).evaluationCompare},
	{Path: "/generation/capabilities", Authentication: routeBearer, Workflow: WorkflowGeneration, WorkflowAction: workflowCapabilitiesAction},
	{Path: "/generation/run", Authentication: routeBearer, Workflow: WorkflowGeneration, WorkflowAction: workflowRunAction},
	{Path: "/training/capabilities", Authentication: routeBearer, Workflow: WorkflowTraining, WorkflowAction: workflowCapabilitiesAction},
	{Path: "/training/run", Authentication: routeBearer, Workflow: WorkflowTraining, WorkflowAction: workflowRunAction},
	{Path: "/model-builder/capabilities", Authentication: routeBearer, Workflow: WorkflowModelBuild, WorkflowAction: workflowCapabilitiesAction},
	{Path: "/model-builder/run", Authentication: routeBearer, Workflow: WorkflowModelBuild, WorkflowAction: workflowRunAction},
	{Path: "/export/capabilities", Authentication: routeBearer, Workflow: WorkflowExport, WorkflowAction: workflowCapabilitiesAction},
	{Path: "/export/run", Authentication: routeBearer, Workflow: WorkflowExport, WorkflowAction: workflowRunAction},
	{Path: "/artifacts", Authentication: routeBearer, Handler: (*Handler).artifactGallery},
	{Path: "/artifacts/content", Authentication: routeBearer, Handler: (*Handler).artifactContent},
	{Path: "/runtime/sessions", Authentication: routeBearer, Handler: (*Handler).runtimeSessions},
	{Path: "/runtime/activity", Authentication: routeBearer, Handler: (*Handler).runtimeActivity},
	{Path: "/runtime/activity/stream", Authentication: routeBearer, Handler: (*Handler).runtimeActivityStream},
	{Path: "/runtime/peers", Authentication: routeBearer, Handler: (*Handler).remotePeerAuthority},
	{Path: "/slots", Authentication: routeBearer, Handler: (*Handler).slotStatus},
	{Path: "/lora-adapters", Authentication: routeBearer, Handler: (*Handler).loraAdapters},
	{Path: "/catalog/models", Authentication: routePublic, Handler: (*Handler).catalogModels},
	{Path: "/hub/search", Authentication: routePublic, Handler: (*Handler).hubSearch},
	{Path: "/hub/downloads", Authentication: routePublic, Handler: (*Handler).hubDownloads},
	{Path: "/agent/tools", Authentication: routePublic, Handler: (*Handler).agentTools},
	{Path: "/agent/step", Authentication: routePublic, Handler: (*Handler).agentStep},
	{Path: "/agent/approval", Authentication: routePublic, Handler: (*Handler).agentApprovalPreview},
	{Path: "/agent/provenance", Authentication: routePublic, Handler: (*Handler).agentProvenance},
	{Path: "/agent/sessions", Authentication: routePublic, Handler: (*Handler).agentSessionList},
}

var routesByPath = compileRouteIndex(routeCatalog)
var versionedRouteFallback = routeDescriptor{Path: "/v1/", Authentication: routeBearer, Handler: (*Handler).serveWebUI}

func compileRouteIndex(routes []routeDescriptor) map[string]routeDescriptor {
	index := make(map[string]routeDescriptor, len(routes))
	for _, route := range routes {
		workflow := route.Workflow != "" && route.WorkflowAction != ""
		if route.Path == "" || (route.Handler == nil) == !workflow || index[route.Path].Path != "" {
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
