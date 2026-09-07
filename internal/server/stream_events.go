package server

// The client's one stream renderer (webui/composer.js) consumes one typed
// event vocabulary, whatever protocol produced the turn: the OpenAI chat
// stream, a complete agent reply, a generation result or a raw media
// body. This file is the server-owned statement of that vocabulary; the
// adapters in composer.js map each served protocol onto it and the
// composer budget test holds the two in step.
const (
	streamEventCreated   = "created"    // the turn identity, before any text; a page notes it to reattach
	streamEventToken     = "token"      // a piece of assistant text
	streamEventToolStart = "tool_start" // a tool call begins: name, arguments
	streamEventToolEnd   = "tool_end"   // the call returned or was refused: result, error, elapsed
	streamEventMedia     = "media"      // an image, video or audio artifact to show in the thread
	streamEventUsage     = "usage"      // token counts and timings for the turn
	streamEventDone      = "done"       // the turn is complete
	streamEventError     = "error"      // the turn failed; a row in the thread, never a banner
)

// streamEventTypes lists the vocabulary in the order the renderer documents it.
var streamEventTypes = []string{
	streamEventCreated, streamEventToken, streamEventToolStart, streamEventToolEnd, streamEventMedia,
	streamEventUsage, streamEventDone, streamEventError,
}
