package server

import "testing"

func TestAdaptiveServingContractMatrix(t *testing.T) {
	contracts := []struct {
		name string
		run  func(*testing.T)
	}{
		{"stream/native", TestNativeCompletionStreaming},
		{"stream/openai-completion", TestStreamingCompletion},
		{"stream/openai-chat", TestStreamingChatCompletion},
		{"stream/openai-responses", TestStreamingResponsesLifecycle},
		{"stream/anthropic-messages", TestStreamingAnthropicMessagesLifecycle},
		{"tools/openai-chat", TestStreamingChatToolCallsAreStructured},
		{"tools/openai-responses", TestStreamingResponsesFunctionCallLifecycle},
		{"tools/anthropic", TestStreamingAnthropicToolUseLifecycle},
		{"embeddings/exact-mixed-input", TestEmbeddingsExactAndMixedTokenInputs},
		{"media/ordered-native", TestNativeCompletionMixedMediaPreservesChunkOrder},
		{"media/ordered-responses", TestResponsesMixedImageAudioPreservesChunkOrder},
		{"media/video", TestChatAndResponsesEncodedVideoProjection},
		{"sessions/continuation", TestResponsesPreviousResponseContinuation},
		{"sessions/tool-call-identity", TestResponsesContinuationRetainsGeneratedToolCallID},
		{"cache/edit-window", TestNativeCompletionNKeepReachesGenerator},
		{"cache/reuse-accounting", TestNativeCompletionPromptCacheAccounting},
		{"cache/projected-identity", TestNativeCompletionProjectedInputsAllowSignedPromptCache},
		{"cancellation/client", TestCancellationPropagates},
		{"cancellation/timeout", TestServerRequestTimeoutCancelsGeneration},
		{"refusal/unbacked-tools", TestResponsesHostedAndCustomToolsUseExplicitPolicy},
		{"refusal/missing-projector", TestNativeCompletionMultimodalRequiresProjector},
	}
	for _, contract := range contracts {
		t.Run(contract.name, contract.run)
	}
}
