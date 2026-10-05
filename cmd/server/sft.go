package main

import (
	"overgo/internal/dataset"
	"overgo/internal/inference"
	"overgo/internal/trainingworkflow"
)

// conversationRenderer renders a conversation as the served model's own chat
// template does: with the generation prompt that invites a reply when asked
// for, and otherwise ending on its last message, as it will have been said.
func conversationRenderer(runner *inference.Runner) trainingworkflow.ConversationRenderer {
	return func(messages []dataset.ChatMessage, generationPrompt bool) (string, error) {
		chat := make([]inference.ChatMessage, len(messages))
		for index, message := range messages {
			chat[index] = inference.ChatMessage{Role: inference.ChatRole(message.Role), Content: message.Content}
		}
		return runner.FormatChatWithOptions(chat, inference.ChatFormatOptions{AddGenerationPrompt: generationPrompt})
	}
}
