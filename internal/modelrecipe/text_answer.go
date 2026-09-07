package modelrecipe

import "overgo/internal/artifact"

// TextAnswerMediaType is the media type of a published text answer.
const TextAnswerMediaType = "text/plain; charset=utf-8"

// TextAnswerContract is the stored form of a text answer (a question about
// an image answered, a transcript's text): the output kind the media
// contracts publish, so the page renders it from the same run. Declared
// with the recipe vocabulary, the one owner the server and the media
// executors already share.
var TextAnswerContract = artifact.DocumentContract{
	Kind: artifact.KindOutput, MediaType: TextAnswerMediaType, Schema: "overgo.text-answer.v1",
}
