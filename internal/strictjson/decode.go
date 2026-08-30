package strictjson

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
)

var ErrTrailingValue = errors.New("trailing JSON value")
var ErrLimit = errors.New("JSON input exceeds size limit")
var ErrDuplicateName = errors.New("duplicate JSON object name")

var null = []byte("null")

func HasValue(data []byte) bool {
	return len(data) != 0 && !bytes.Equal(data, null)
}

func Decode(reader io.Reader, destination any) error {
	data, err := io.ReadAll(reader)
	if err != nil {
		return err
	}
	return DecodeBytes(data, destination)
}

// rejectDuplicateNames walks the raw document bytes directly: values are
// skipped span-wise without re-materializing them, and only object names —
// the small part — are decoded for the duplicate check. The former
// re-tokenizing pass allocated every value a second time; this walker is its
// replacement, not an addition.
func rejectDuplicateNames(data []byte) error {
	walker := duplicateNameWalker{data: data}
	if err := walker.value(); err != nil {
		return err
	}
	return nil
}

type duplicateNameWalker struct {
	data []byte
	at   int
}

func (w *duplicateNameWalker) skipSpace() {
	for w.at < len(w.data) {
		switch w.data[w.at] {
		case ' ', '\t', '\r', '\n':
			w.at++
		default:
			return
		}
	}
}

func (w *duplicateNameWalker) value() error {
	w.skipSpace()
	if w.at >= len(w.data) {
		return errors.New("JSON value is truncated")
	}
	switch w.data[w.at] {
	case '{':
		return w.object()
	case '[':
		return w.array()
	case '"':
		_, err := w.stringSpan()
		return err
	default:
		return w.scalar()
	}
}

func (w *duplicateNameWalker) object() error {
	w.at++
	names := map[string]struct{}{}
	w.skipSpace()
	if w.at < len(w.data) && w.data[w.at] == '}' {
		w.at++
		return nil
	}
	for {
		w.skipSpace()
		name, err := w.stringSpan()
		if err != nil {
			return errors.New("JSON object name is invalid")
		}
		key := string(name)
		if bytes.IndexByte(name, '\\') >= 0 {
			// A differently-escaped spelling of one name is still one name:
			// only escaped names pay the normalization allocation.
			quoted := make([]byte, 0, len(name)+2)
			quoted = append(append(append(quoted, '"'), name...), '"')
			if err := json.Unmarshal(quoted, &key); err != nil {
				return errors.New("JSON object name is invalid")
			}
		}
		if _, duplicate := names[key]; duplicate {
			return ErrDuplicateName
		}
		names[key] = struct{}{}
		w.skipSpace()
		if w.at >= len(w.data) || w.data[w.at] != ':' {
			return errors.New("JSON object member is invalid")
		}
		w.at++
		if err := w.value(); err != nil {
			return err
		}
		w.skipSpace()
		if w.at >= len(w.data) {
			return errors.New("JSON object is truncated")
		}
		switch w.data[w.at] {
		case ',':
			w.at++
		case '}':
			w.at++
			return nil
		default:
			return errors.New("JSON object is invalid")
		}
	}
}

func (w *duplicateNameWalker) array() error {
	w.at++
	w.skipSpace()
	if w.at < len(w.data) && w.data[w.at] == ']' {
		w.at++
		return nil
	}
	for {
		if err := w.value(); err != nil {
			return err
		}
		w.skipSpace()
		if w.at >= len(w.data) {
			return errors.New("JSON array is truncated")
		}
		switch w.data[w.at] {
		case ',':
			w.at++
		case ']':
			w.at++
			return nil
		default:
			return errors.New("JSON array is invalid")
		}
	}
}

// stringSpan skips one JSON string, returning its raw contents without
// unescaping. Escaped names compare by raw span, which is exact for the
// duplicate check because encoders emit one canonical escaping per name and
// the strict decoder has already proven the document well-formed.
func (w *duplicateNameWalker) stringSpan() ([]byte, error) {
	if w.at >= len(w.data) || w.data[w.at] != '"' {
		return nil, errors.New("JSON string is invalid")
	}
	w.at++
	start := w.at
	for w.at < len(w.data) {
		switch w.data[w.at] {
		case '\\':
			w.at += 2
		case '"':
			span := w.data[start:w.at]
			w.at++
			return span, nil
		default:
			w.at++
		}
	}
	return nil, errors.New("JSON string is truncated")
}

func (w *duplicateNameWalker) scalar() error {
	start := w.at
	for w.at < len(w.data) {
		switch w.data[w.at] {
		case ',', '}', ']', ' ', '\t', '\r', '\n':
			if w.at == start {
				return errors.New("JSON scalar is invalid")
			}
			return nil
		default:
			w.at++
		}
	}
	if w.at == start {
		return errors.New("JSON scalar is invalid")
	}
	return nil
}

// DecodeBytes decodes one complete strict document from bytes the caller
// already holds: the duplicate-name scan re-reads that same slice, so the
// payload is never copied into a scratch buffer on the way through.
func DecodeBytes(data []byte, destination any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); errors.Is(err, io.EOF) {
		return rejectDuplicateNames(data)
	} else if err != nil {
		return err
	}
	return ErrTrailingValue
}

func DecodeBounded(reader io.Reader, limit int64, destination any) error {
	limited := &io.LimitedReader{R: reader, N: limit + 1}
	err := Decode(limited, destination)
	if limited.N == 0 {
		return ErrLimit
	}
	return err
}
