package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
)

func serveTestRequest(
	handler http.Handler,
	method, path, body string,
) *httptest.ResponseRecorder {
	request := httptest.NewRequest(method, path, strings.NewReader(body))
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}
