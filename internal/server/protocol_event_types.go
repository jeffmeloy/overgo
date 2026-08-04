package server

type responsesStreamEvent struct {
	Arguments    *string `json:"arguments,omitempty"`
	Delta        any     `json:"delta,omitempty"`
	Item         any     `json:"item,omitempty"`
	ItemID       string  `json:"item_id,omitempty"`
	OutputIndex  *int    `json:"output_index,omitempty"`
	Part         any     `json:"part,omitempty"`
	Response     any     `json:"response,omitempty"`
	ResponseID   string  `json:"response_id,omitempty"`
	SummaryIndex *int    `json:"summary_index,omitempty"`
	Text         *string `json:"text,omitempty"`
	Type         string  `json:"type"`
}

type responsesProgress struct {
	ID     string `json:"id"`
	Object string `json:"object"`
	Status string `json:"status"`
}

type responsesMessageStart struct {
	Content []any  `json:"content"`
	ID      string `json:"id"`
	Role    string `json:"role"`
	Status  string `json:"status"`
	Type    string `json:"type"`
}

type responsesTextStart struct {
	Text string `json:"text"`
	Type string `json:"type"`
}

type anthropicStreamEvent struct {
	ContentBlock any    `json:"content_block,omitempty"`
	Delta        any    `json:"delta,omitempty"`
	Index        *int   `json:"index,omitempty"`
	Message      any    `json:"message,omitempty"`
	Type         string `json:"type"`
	Usage        any    `json:"usage,omitempty"`
}

type anthropicMessageStart struct {
	Content      []any          `json:"content"`
	ID           string         `json:"id"`
	Model        string         `json:"model"`
	Role         string         `json:"role"`
	StopReason   any            `json:"stop_reason"`
	StopSequence any            `json:"stop_sequence"`
	Type         string         `json:"type"`
	Usage        anthropicUsage `json:"usage"`
}

type anthropicContentBlockStart struct {
	ID        string          `json:"id,omitempty"`
	Input     *map[string]any `json:"input,omitempty"`
	Name      string          `json:"name,omitempty"`
	Signature *string         `json:"signature,omitempty"`
	Text      *string         `json:"text,omitempty"`
	Thinking  *string         `json:"thinking,omitempty"`
	Type      string          `json:"type"`
}

type anthropicContentDelta struct {
	PartialJSON *string `json:"partial_json,omitempty"`
	Signature   *string `json:"signature,omitempty"`
	Text        *string `json:"text,omitempty"`
	Thinking    *string `json:"thinking,omitempty"`
	Type        string  `json:"type"`
}

type anthropicMessageDelta struct {
	StopReason   string `json:"stop_reason"`
	StopSequence any    `json:"stop_sequence"`
}

type anthropicOutputUsage struct {
	OutputTokens int `json:"output_tokens"`
}

func eventIndex(index int) *int {
	return &index
}

func eventString(value string) *string {
	return &value
}

func emptyEventObject() *map[string]any {
	value := map[string]any{}
	return &value
}
