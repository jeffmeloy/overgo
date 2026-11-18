// Package enhancerchat opens the chat model a picture model's folder serves beside its weights, as MOSS's prompt
// enhancer, apart from the picture package: a chat links the language runtime, and a picture package that linked it
// linked it into every program drawing a picture (simplediffusion-peak: the SimpleDiffusion test program grew from 31 to
// 40 MB and its process peak lost its lead over the adaptive reference's).
package enhancerchat

import "context"

// Chat answers request under system with at most maxTokens of the text it samples.
type Chat func(ctx context.Context, system, request string, maxTokens int) (string, error)
