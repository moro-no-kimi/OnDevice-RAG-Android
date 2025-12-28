package com.ml.shubham0204.docqa.data

import android.content.Context
import android.content.SharedPreferences
import org.koin.core.annotation.Single

@Single
class AppSettings(
    context: Context,
) {
    private val prefsFileName = "app_settings"
    private val rerankerEnabledKey = "reranker_enabled"

    private val sharedPreferences: SharedPreferences =
        context.getSharedPreferences(prefsFileName, Context.MODE_PRIVATE)

    fun setRerankerEnabled(enabled: Boolean) {
        sharedPreferences.edit().putBoolean(rerankerEnabledKey, enabled).apply()
    }

    fun isRerankerEnabled(): Boolean = sharedPreferences.getBoolean(rerankerEnabledKey, true)
}
