package gpt2

// #cgo CFLAGS: -I./ggml.cpp/include/ggml/ -I./ggml.cpp/examples/ -I./ggml.cpp/src/
// #cgo CXXFLAGS: -I./ggml.cpp/include/ggml/ -I./ggml.cpp/examples/ -I./ggml.cpp/src/
// #cgo darwin LDFLAGS: -framework Accelerate
// #cgo LDFLAGS: -lgpt2 -lm -lstdc++
// #include <gpt2.h>
import "C"
import (
	"fmt"
	"strings"
	"unsafe"
)

type gpt2 struct {
	state unsafe.Pointer
}

func New(model string) (*gpt2, error) {
	state := C.gpt2_allocate_state()
	modelPath := C.CString(model)
	result := C.gpt2_bootstrap(modelPath, state)
	if result != 0 {
		return nil, fmt.Errorf("failed loading model")
	}

	return &gpt2{state: state}, nil
}

func (l *gpt2) Predict(text string, opts ...PredictOption) (string, error) {

	po := NewPredictOptions(opts...)

	input := C.CString(text)
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}
	out := make([]byte, po.Tokens)

	params := C.gpt2_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.int(po.Batch))
	ret := C.gpt2_predict(params, l.state, (*C.char)(unsafe.Pointer(&out[0])))
	if ret != 0 {
		return "", fmt.Errorf("inference failed")
	}
	res := C.GoString((*C.char)(unsafe.Pointer(&out[0])))

	res = strings.TrimPrefix(res, " ")
	res = strings.TrimPrefix(res, text)
	res = strings.TrimPrefix(res, "\n")
	res = strings.TrimSuffix(res, "<|endoftext|>")
	C.gpt2_free_params(params)

	return res, nil
}

func (l *gpt2) Free() {
	C.gpt2_free_model(l.state)
}
