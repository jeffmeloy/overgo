package strictjson

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
)

var ErrTrailingValue = errors.New("trailing JSON value")
var ErrLimit = errors.New("JSON input exceeds size limit")

func Decode(reader io.Reader, destination any) error {
	decoder := json.NewDecoder(reader)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); errors.Is(err, io.EOF) {
		return nil
	} else if err != nil {
		return err
	}
	return ErrTrailingValue
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
