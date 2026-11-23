package latentimage

import (
	"bytes"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"slices"
)

// The keywords a picture's stored bytes carry what it was drawn from under, each an international text chunk (iTXt)
// of UTF-8, as the MOSS Ultra workflow shows them beside its pictures, and its finishing.
const (
	textPromptUsed     = "Prompt used"
	textNegativeUsed   = "Negative prompt used"
	textFinishing      = "Finishing"
	finishingSharpened = "sharpened"
)

// pngSignature opens every PNG; pngChunkFrame is a chunk's length, type and CRC around its data.
var pngSignature = []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}

const (
	pngChunkLength = 4
	pngChunkType   = 4
	pngChunkCRC    = 4
	pngTextChunk   = "iTXt"
)

// pngChunk is one chunk of a PNG: its type, its data, and where it lies in the file.
type pngChunk struct {
	kind       string
	data       []byte
	start, end int
}

// pngChunks are a PNG's chunks in order, or why the bytes are no PNG.
func pngChunks(data []byte) ([]pngChunk, error) {
	if !bytes.HasPrefix(data, pngSignature) {
		return nil, errors.New("latent image: the picture is no PNG")
	}
	var chunks []pngChunk
	for at := len(pngSignature); at < len(data); {
		if len(data)-at < pngChunkLength+pngChunkType+pngChunkCRC {
			return nil, errors.New("latent image: a PNG chunk is cut short")
		}
		length := int(binary.BigEndian.Uint32(data[at:]))
		end := at + pngChunkLength + pngChunkType + length + pngChunkCRC
		if length < 0 || end > len(data) || end < at {
			return nil, errors.New("latent image: a PNG chunk runs past the picture")
		}
		body := at + pngChunkLength + pngChunkType
		chunks = append(chunks, pngChunk{kind: string(data[at+pngChunkLength : body]), data: data[body : body+length], start: at, end: end})
		at = end
	}
	return chunks, nil
}

// textChunkKeyword and textChunkValue are an uncompressed iTXt chunk's keyword and text: keyword, NUL, compression flag
// and method, language tag, NUL, translated keyword, NUL, text.
func textChunkKeyword(data []byte) (string, string, bool) {
	keyword, rest, found := bytes.Cut(data, []byte{0})
	if !found || len(rest) < 2 || rest[0] != 0 {
		return "", "", false
	}
	_, rest, found = bytes.Cut(rest[2:], []byte{0})
	if !found {
		return "", "", false
	}
	_, text, found := bytes.Cut(rest, []byte{0})
	return string(keyword), string(text), found
}

// pictureTextKeywords are the keywords a picture's texts are written under, which a picture written again replaces.
var pictureTextKeywords = []string{textPromptUsed, textNegativeUsed, textFinishing}

// withPictureTexts is a PNG with its picture texts replaced by texts, each written after its header (IHDR) in the order
// given: the bytes as they were where it held none and is given none.
func withPictureTexts(data []byte, texts [][2]string) ([]byte, error) {
	chunks, err := pngChunks(data)
	if err != nil {
		return nil, err
	}
	held := slices.ContainsFunc(chunks, func(chunk pngChunk) bool {
		keyword, _, ok := textChunkKeyword(chunk.data)
		return chunk.kind == pngTextChunk && ok && slices.Contains(pictureTextKeywords, keyword)
	})
	if !held && len(texts) == 0 {
		return data, nil
	}
	if len(chunks) == 0 || chunks[0].kind != "IHDR" {
		return nil, errors.New("latent image: a PNG opens with no header")
	}
	out := bytes.NewBuffer(make([]byte, 0, len(data)))
	out.Write(data[:chunks[0].end])
	for _, text := range texts {
		body := append(append([]byte(text[0]), 0, 0, 0, 0, 0), []byte(text[1])...)
		var frame [pngChunkLength]byte
		binary.BigEndian.PutUint32(frame[:], uint32(len(body)))
		out.Write(frame[:])
		typed := append([]byte(pngTextChunk), body...)
		out.Write(typed)
		binary.BigEndian.PutUint32(frame[:], crc32.ChecksumIEEE(typed))
		out.Write(frame[:])
	}
	for _, chunk := range chunks[1:] {
		if keyword, _, ok := textChunkKeyword(chunk.data); chunk.kind == pngTextChunk && ok && slices.Contains(pictureTextKeywords, keyword) {
			continue
		}
		out.Write(data[chunk.start:chunk.end])
	}
	return out.Bytes(), nil
}

// readPictureTexts are a PNG's picture texts by keyword, none where it holds none.
func readPictureTexts(data []byte) map[string]string {
	chunks, err := pngChunks(data)
	if err != nil {
		return nil
	}
	texts := map[string]string{}
	for _, chunk := range chunks {
		if keyword, text, ok := textChunkKeyword(chunk.data); chunk.kind == pngTextChunk && ok && slices.Contains(pictureTextKeywords, keyword) {
			texts[keyword] = text
		}
	}
	return texts
}
