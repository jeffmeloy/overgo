package inference

import (
	"errors"
	"fmt"
	"strings"
	"time"
)

func chatTemplateRaiseException(message string) (string, error) {
	return "", errors.New(message)
}

func newChatTemplateStrftime(now time.Time) func(string) (string, error) {
	return func(format string) (string, error) {
		result, err := formatChatTemplateTime(now, format)
		if err != nil {
			return "", err
		}
		return result, nil
	}
}

func formatChatTemplateTime(value time.Time, format string) (string, error) {
	var output strings.Builder
	for index := 0; index < len(format); index++ {
		if format[index] != '%' {
			output.WriteByte(format[index])
			continue
		}
		index++
		if index >= len(format) {
			return "", errors.New("strftime_now format ends with %")
		}
		var rendered string
		switch format[index] {
		case '%':
			rendered = "%"
		case 'a':
			rendered = value.Format("Mon")
		case 'A':
			rendered = value.Format("Monday")
		case 'b', 'h':
			rendered = value.Format("Jan")
		case 'B':
			rendered = value.Format("January")
		case 'c':
			rendered = value.Format("Mon Jan 02 15:04:05 2006")
		case 'd':
			rendered = value.Format("02")
		case 'e':
			rendered = fmt.Sprintf("%2d", value.Day())
		case 'H':
			rendered = value.Format("15")
		case 'I':
			rendered = value.Format("03")
		case 'j':
			rendered = fmt.Sprintf("%03d", value.YearDay())
		case 'm':
			rendered = value.Format("01")
		case 'M':
			rendered = value.Format("04")
		case 'p':
			rendered = value.Format("PM")
		case 'S':
			rendered = value.Format("05")
		case 'u':
			weekday := int(value.Weekday())
			if weekday == 0 {
				weekday = 7
			}
			rendered = fmt.Sprint(weekday)
		case 'w':
			rendered = fmt.Sprint(int(value.Weekday()))
		case 'x':
			rendered = value.Format("01/02/06")
		case 'X':
			rendered = value.Format("15:04:05")
		case 'y':
			rendered = value.Format("06")
		case 'Y':
			rendered = value.Format("2006")
		case 'z':
			rendered = value.Format("-0700")
		case 'Z':
			rendered = value.Format("MST")
		default:
			return "", fmt.Errorf(
				"strftime_now directive %%%c is unsupported",
				format[index],
			)
		}
		output.WriteString(rendered)
	}
	return output.String(), nil
}
