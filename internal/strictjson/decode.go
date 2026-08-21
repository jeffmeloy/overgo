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
	var source bytes.Buffer
	decoder := json.NewDecoder(io.TeeReader(reader, &source))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); errors.Is(err, io.EOF) {
		return rejectDuplicateNames(source.Bytes())
	} else if err != nil {
		return err
	}
	return ErrTrailingValue
}

func rejectDuplicateNames(data []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	var scan func() error
	scan = func() error {
		token, err := decoder.Token()
		if err != nil {
			return err
		}
		delimiter, composite := token.(json.Delim)
		if !composite {
			return nil
		}
		switch delimiter {
		case '{':
			names := make(map[string]struct{})
			for decoder.More() {
				key, err := decoder.Token()
				if err != nil {
					return err
				}
				name, ok := key.(string)
				if !ok {
					return errors.New("JSON object name is invalid")
				}
				if _, duplicate := names[name]; duplicate {
					return ErrDuplicateName
				}
				names[name] = struct{}{}
				if err := scan(); err != nil {
					return err
				}
			}
		case '[':
			for decoder.More() {
				if err := scan(); err != nil {
					return err
				}
			}
		default:
			return errors.New("JSON delimiter is invalid")
		}
		_, err = decoder.Token()
		return err
	}
	return scan()
}

func DecodeBytes(data []byte, destination any) error {
	return Decode(bytes.NewReader(data), destination)
}

func DecodeBounded(reader io.Reader, limit int64, destination any) error {
	limited := &io.LimitedReader{R: reader, N: limit + 1}
	err := Decode(limited, destination)
	if limited.N == 0 {
		return ErrLimit
	}
	return err
}
