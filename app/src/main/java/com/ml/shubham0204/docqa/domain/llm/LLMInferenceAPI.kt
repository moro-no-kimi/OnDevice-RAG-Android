package com.ml.shubham0204.docqa.domain.llm

import kotlinx.coroutines.flow.Flow

abstract class LLMInferenceAPI {
    abstract suspend fun getResponse(prompt: String): String?
    
    abstract fun getResponseStream(prompt: String): Flow<String>
}
