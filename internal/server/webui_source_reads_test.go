package server

import (
	"testing"
)

// webuiSourceClass is why a test reads the served UI source instead of
// driving the page: a ratchet (a count ceiling, a census or a one-owner
// rule), a ban (a pattern the source must not hold) or a contract (how the
// server serves the page).
type webuiSourceClass string

const (
	sourceRatchet  webuiSourceClass = "ratchet"
	sourceBan      webuiSourceClass = "ban"
	sourceContract webuiSourceClass = "contract"
)

// webuiSourceReaders declares every server test that reads the served UI
// source; internal/gate's census holds the declaration to the tests.
var webuiSourceReaders = map[string]webuiSourceClass{
	"TestWebUIComposerBudget":               sourceRatchet,
	"TestWebUIReviewRatchet":                sourceRatchet,
	"TestArtifactContentURLHasOneOwner":     sourceRatchet,
	"TestWebUIClientHelpersHaveOneOwner":    sourceRatchet,
	"TestWebUIRouteTable":                   sourceRatchet,
	"TestAgentRoutesHaveOnePrefix":          sourceRatchet,
	"TestWebUIClientDedup":                  sourceRatchet,
	"TestWebUIStyleInvariants":              sourceRatchet,
	"TestGUITrainingAndExportShareServices": sourceRatchet,

	"TestWebUIOneShell":                         sourceContract,
	"TestWebUIServesEmbeddedAssets":             sourceContract,
	"TestWebUIContentSecurityPolicy":            sourceContract,
	"TestWebUIAssetsPublicWhenAPIKeyConfigured": sourceContract,
	"TestWebUIUnknownPathReturns404":            sourceContract,
	"TestWebUIDevelopmentDirectory":             sourceContract,
	"TestIdleShellAnswersWhileNothingServes":    sourceContract,
	"TestGUIMobileDeviceEvidence":               sourceContract,
	"TestGUIMobileEvidenceContract":             sourceContract,

	"TestFrontPageConversations":                                  sourceBan,
	"TestWorkspaceCapabilityDocument":                             sourceBan,
	"TestArtifactGalleryProjectedPageIsBoundedAndStreamsPayloads": sourceBan,
	"TestGenerationWorkspaceUsesRecipeCapabilities":               sourceBan,
	"TestGlobalOperationShell":                                    sourceBan,
	"TestRuntimeActivitySessionLedgerGUI":                         sourceBan,
	"TestWebUITabsLoadThroughOneResource":                         sourceBan,
	"TestWebUIWorkspaceRoutes":                                    sourceBan,
	"TestWebUIRuntimeMonitor":                                     sourceBan,
	"TestAgentGUIUsesProjectedQueriesAndSSE":                      sourceBan,
	"TestWebUIChatUsesServerContextAndTiming":                     sourceBan,
	"TestWebUIChatMarkdown":                                       sourceBan,
	"TestRLWorkspaceUsesGenericWorkflowEndpoints":                 sourceBan,
	"TestPeerWorkspaceUsesGlobalOperations":                       sourceBan,

	"TestActiveRecipeInspectorUsesCompiledOrder":      sourceBan,
	"TestEvaluationWorkbenchUsesDeclaredCapabilities": sourceBan,
	"TestRLWorkspaceRendersMeasuredEvidence":          sourceBan,

	"TestAgentWorkspaceNoHiddenReasoning": sourceBan,

	"TestFrontPageModelSwitch":            sourceRatchet,
	"TestFrontPageAgent":                  sourceBan,
	"TestFrontPageGenerationDeclarations": sourceBan,
}

// TestWebUITestsDriveBehaviourNotSource holds the tests that read the
// served UI source to a declared reason: each is a ratchet, a ban or a
// serving contract; behaviour is driven by a browser leg. internal/gate's
// TestServerTestsDeclareEveryWebUISourceRead holds the declaration to the
// tests themselves.
func TestWebUITestsDriveBehaviourNotSource(t *testing.T) {
	t.Parallel()
	for test, class := range webuiSourceReaders {
		if class != sourceRatchet && class != sourceBan && class != sourceContract {
			t.Errorf("%s reads the served source for an undeclared reason %q", test, class)
		}
	}
}
