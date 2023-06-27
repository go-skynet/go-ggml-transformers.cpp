package gpt2

import (
	common "github.com/go-skynet/go-common"
)

// var DefaultModelInitializationOptions common.InitializationOptions = common.InitializationOptions{}
// var MergeInitializationOptionsWithDefaults = common.GetMergeInitializationOptionsFnFromDefault(DefaultModelInitializationOptions)

var DefaultPredictOptions common.PredictTextOptions = common.PredictTextOptions{
	Seed:        -1,
	Threads:     4,
	Tokens:      200,
	TopK:        40,
	TopP:        0.90,
	Temperature: 0.96,
	Batch:       9,
}

var MergePredictOptionsWithDefaults = common.GetMergePredictTextOptionsFnFromDefault(DefaultPredictOptions)
